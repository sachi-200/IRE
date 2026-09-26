"""MIND-large v2 reranker: ONE feature function shared by training
(train_mind_v2.py, on labelled MINDlarge_dev) and the Codabench submission
(generate_mind_v2_submission.py, on unlabelled MINDlarge_test), so the model
is scored on exactly the feature distribution it was trained on.

Every feature is computable on the unlabelled test set -- nothing reads a
click label from the file being scored:
  - retrieval: BM25 + semantic score from the last 10 history items (same
    as the v1 submission) and a longer semantic profile (last 50 items);
  - user: full history length, category/subcategory share of the history;
  - article priors: clicks/impressions/CTR from EARLIER labelled files
    (train for training on dev; train+dev for scoring test);
  - in-view popularity: how often the article was SHOWN in the previous
    1/6/24 full hours, counted over all behaviors files up to and including
    the one being scored (exposure only, no labels), plus hours since the
    article was first shown;
  - each score re-expressed relative to the impression's other candidates.

The v1 features (src.reranker.FEATURE_COLS, exactly as
generate_mind_reranked_submission.py computes them) are also produced, so
train_mind_v2.py can score the currently-submitted model on the same
holdout for a like-for-like comparison.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ann_index import ANNIndex
from src.bm25_index import BM25Index
from src.config import COLD_START_MAX_CLICKS
from src.embeddings import compute_embeddings, embeddings_exist, load_embeddings, save_embeddings
from src.query_builder import build_query, build_user_embedding
from src.reranker import FEATURE_COLS, add_impression_relative_features

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
BEH_DTYPES = {"impression_id": str, "user_id": str, "time": str, "history": str, "impressions": str}
TIME_FMT = "%m/%d/%Y %I:%M:%S %p"
CHUNK_SIZE = 10_000
N_RECENT = 10        # v1 submission's history window (BM25 query + semantic vector)
N_RECENT_LONG = 50   # longer profile for semantic_score_long and category shares
INVIEW_WINDOWS_HOURS = (1, 6, 24)
_HOUR_BLOCK = 10**6  # > any "hours since epoch" value, see InviewCounts

V2_RELATIVE_SOURCE_COLS = ["semantic_score", "semantic_score_long", "bm25_score", "prior_ctr",
                           "inview_prev_1h", "inview_prev_24h", "hours_since_first_seen"]
V2_FEATURE_COLS = (
    ["semantic_score", "semantic_score_long", "bm25_score",
     "n_hist", "cat_frac_10", "cat_frac_50", "subcat_frac_50",
     "prior_clicks", "prior_impressions", "prior_ctr"]
    + [f"inview_prev_{h}h" for h in INVIEW_WINDOWS_HOURS]
    + ["hours_since_first_seen", "n_candidates"]
    + [f"{c}_{kind}" for c in V2_RELATIVE_SOURCE_COLS for kind in ("pct_rank", "z", "gap")]
)


def iter_behaviors(path: Path, chunksize: int = CHUNK_SIZE, usecols=None):
    return pd.read_csv(path, sep="\t", header=None, names=BEH_COLS, usecols=usecols,
                       dtype=BEH_DTYPES, chunksize=chunksize)


def load_news(dirs) -> pd.DataFrame:
    """Same corpus as generate_mind_reranked_submission.py: train+dev+test
    news.tsv, deduplicated in that order, text = title + abstract. Keeping
    it identical matters -- BM25's IDF depends on the corpus."""
    news = pd.concat([pd.read_csv(Path(d) / "news.tsv", sep="\t", header=None, names=NEWS_COLS)
                      for d in dirs], ignore_index=True).drop_duplicates("news_id")
    news["text"] = (news["title"].fillna("") + " " + news["abstract"].fillna("")).str.strip()
    return news.reset_index(drop=True)


class InviewCounts:
    """How many times each article was shown, per (article, hour). Window
    counts over [h - w, h) (strictly earlier FULL hours) are two
    np.searchsorted calls on one sorted int64 axis -- key = code * block +
    hour -- instead of a per-article loop."""

    def __init__(self, agg: pd.Series):
        codes = agg.index.get_level_values(0).to_numpy(np.int64)
        hours = agg.index.get_level_values(1).to_numpy(np.int64)
        key = codes * _HOUR_BLOCK + hours
        order = np.argsort(key, kind="stable")
        self.keys = key[order]
        self.prefix = np.concatenate([[0], np.cumsum(agg.to_numpy(np.int64)[order])])

    def window(self, codes: np.ndarray, hours: np.ndarray, w: int) -> np.ndarray:
        base = codes.astype(np.int64) * _HOUR_BLOCK
        hi = np.searchsorted(self.keys, base + hours, side="left")
        lo = np.searchsorted(self.keys, base + hours - w, side="left")
        return (self.prefix[hi] - self.prefix[lo]).astype(np.float32)


@dataclass
class MindContext:
    bm25: BM25Index
    ann: ANNIndex
    article_text: dict
    id_to_embedding: dict
    art_code: dict
    cat_of_code: np.ndarray
    sub_of_code: np.ndarray
    n_cats: int
    n_subs: int
    prior_clicks: np.ndarray
    prior_impr: np.ndarray
    pos_ctr_prior: dict
    inview: InviewCounts
    first_seen: np.ndarray


def _to_seconds(times: pd.Series) -> np.ndarray:
    return (pd.to_datetime(times, format=TIME_FMT).astype("int64") // 10**9).to_numpy()


def scan_behaviors(path: Path, art_code: dict, n_codes: int, use_labels: bool, use_inview: bool):
    """One streaming pass over a behaviors.tsv. use_labels -> accumulate
    article clicks/impressions and position CTR (only for files strictly
    BEFORE the one being scored). use_inview -> accumulate label-free
    exposure counts per (article, hour) and each article's first-shown time."""
    clicks = np.zeros(n_codes, np.int64)
    impr = np.zeros(n_codes, np.int64)
    first_seen = np.full(n_codes, np.inf)
    pos_clicks, pos_impr = {}, {}
    inview_parts = []
    print(f"  scanning {path} (labels={use_labels}, in-view={use_inview})...")
    for chunk in iter_behaviors(path, usecols=["time", "impressions"]):
        chunk = chunk.dropna(subset=["impressions"])
        if chunk.empty:
            continue
        secs = pd.Series(_to_seconds(chunk["time"]), index=chunk.index)
        tokens = chunk["impressions"].str.split().explode().dropna()
        parts = tokens.str.rsplit("-", n=1, expand=True)
        codes = parts[0].map(art_code)
        known = codes.notna().to_numpy()
        code_arr = codes[known].astype(np.int64).to_numpy()
        tok_secs = secs.loc[tokens.index].to_numpy()[known]

        if use_inview:
            hours = tok_secs // 3600
            inview_parts.append(pd.DataFrame({"code": code_arr, "hour": hours}).value_counts())
            np.minimum.at(first_seen, code_arr, tok_secs.astype(float))
        if use_labels and parts.shape[1] > 1:
            labels = pd.to_numeric(parts[1], errors="coerce")
            ok = known & labels.notna().to_numpy()
            lab = labels.to_numpy()[ok].astype(np.int64)
            c = codes.to_numpy()[ok].astype(np.int64)
            np.add.at(clicks, c, lab)
            np.add.at(impr, c, 1)
            pos = tokens.groupby(level=0).cumcount().to_numpy()[ok]
            gp =pd.DataFrame({"pos": pos, "lab": lab}).groupby("pos")["lab"].agg(["sum", "count"])
            for p, row in gp.iterrows():
                pos_clicks[p] = pos_clicks.get(p, 0) + int(row["sum"])
                pos_impr[p] = pos_impr.get(p, 0) + int(row["count"])
    inview = (pd.concat(inview_parts).groupby(level=[0, 1]).sum() if inview_parts
              else pd.Series(dtype=np.int64))
    return {"clicks": clicks, "impr": impr, "first_seen": first_seen,
            "pos_clicks": pos_clicks, "pos_impr": pos_impr, "inview": inview}


def build_context(news_dirs, prior_files, inview_files, embeddings_name: str = "mind_large") -> MindContext:
    news = load_news(news_dirs)
    print(f"  corpus: {len(news)} unique articles")
    ids = news["news_id"].tolist()
    art_code = {a: i for i, a in enumerate(ids)}
    cat_codes, cats = pd.factorize(news["category"])
    sub_codes, subs = pd.factorize(news["subcategory"])

    print("  building BM25 + semantic indexes...")
    bm25 = BM25Index().fit(ids, news["text"].tolist())
    if embeddings_exist(embeddings_name):
        emb_ids, emb = load_embeddings(embeddings_name)
    else:
        print(f"  computing embeddings_{embeddings_name}.npz (one-time, cached after)...")
        emb_ids, emb = compute_embeddings(ids, news["text"].tolist())
        save_embeddings(embeddings_name, emb_ids, emb)
    ann = ANNIndex().fit(emb_ids, emb)

    n = len(ids)
    clicks, impr = np.zeros(n, np.int64), np.zeros(n, np.int64)
    first_seen = np.full(n, np.inf)
    pos_clicks, pos_impr, inview_parts = {}, {}, []
    prior_set = {str(p) for p in prior_files}
    inview_set = {str(p) for p in inview_files}
    for path in dict.fromkeys(list(map(str, prior_files)) + list(map(str, inview_files))):
        s = scan_behaviors(Path(path), art_code, n, path in prior_set, path in inview_set)
        clicks += s["clicks"]; impr += s["impr"]
        first_seen = np.minimum(first_seen, s["first_seen"])
        for p, v in s["pos_clicks"].items():
            pos_clicks[p] = pos_clicks.get(p, 0) + v
        for p, v in s["pos_impr"].items():
            pos_impr[p] = pos_impr.get(p, 0) + v
        if len(s["inview"]):
            inview_parts.append(s["inview"])
    inview = pd.concat(inview_parts).groupby(level=[0, 1]).sum()
    print(f"  priors: {int((impr > 0).sum())} articles with labelled history; "
          f"in-view: {len(inview)} (article, hour) buckets")

    return MindContext(
        bm25=bm25, ann=ann, article_text=dict(zip(ids, news["text"])),
        id_to_embedding=dict(zip(emb_ids, emb)), art_code=art_code,
        cat_of_code=cat_codes.astype(np.int64), sub_of_code=sub_codes.astype(np.int64),
        n_cats=len(cats) + 1, n_subs=len(subs) + 1,
        prior_clicks=clicks, prior_impr=impr,
        pos_ctr_prior={p: pos_clicks[p] / pos_impr[p] for p in pos_impr},
        inview=InviewCounts(inview), first_seen=first_seen)


def _share_lookup(hist_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    """count of each query key among hist_keys (0 if absent)."""
    if len(hist_keys) == 0:
        return np.zeros(len(query_keys))
    uk, uc = np.unique(hist_keys, return_counts=True)
    idx = np.clip(np.searchsorted(uk, query_keys), 0, len(uk) - 1)
    return np.where(uk[idx] == query_keys, uc[idx], 0).astype(float)


def chunk_candidates(chunk: pd.DataFrame, ctx: MindContext, labeled: bool):
    """chunk: raw behaviors rows. Returns (cand, impression_ids, n_per_impression)
    -- cand has one row per candidate in ORIGINAL order, with V2_FEATURE_COLS,
    the v1 FEATURE_COLS, `impression_id` (row index within the chunk) and
    `clicked` (labeled=True only)."""
    n = len(chunk)
    imp_ids = chunk["impression_id"].to_numpy()
    hists = chunk["history"].to_numpy()
    imps = chunk["impressions"].to_numpy()
    secs = _to_seconds(chunk["time"])

    counts = np.zeros(n, np.int64)
    n_hist = np.zeros(n)
    arts, labels, bm, sem, sem_long = [], [], [], [], []
    hist_imp, hist_code, hist_in10 = [], [], []
    for r in range(n):
        toks = imps[r].split() if isinstance(imps[r], str) else []
        if labeled:
            split = [t.rsplit("-", 1) for t in toks]
            toks = [s[0] for s in split if len(s) == 2]
            labels.extend(int(s[1]) for s in split if len(s) == 2)
        counts[r] = len(toks)
        hist = hists[r].split() if isinstance(hists[r], str) else []
        n_hist[r] = len(hist)
        recent, recent_long = hist[-N_RECENT:], hist[-N_RECENT_LONG:]
        for j, a in enumerate(recent_long):
            c = ctx.art_code.get(a)
            if c is not None:
                hist_imp.append(r); hist_code.append(c)
                hist_in10.append(j >= len(recent_long) - len(recent))
        if not toks:
            continue
        arts.extend(toks)

        query = build_query(recent, ctx.article_text, n_recent=N_RECENT)
        d = ctx.bm25.score_docs(toks, query) if query else {}
        bm.extend(d.get(t, 0.0) for t in toks)
        v = build_user_embedding(recent, ctx.id_to_embedding, n_recent=N_RECENT)
        sem.extend(ctx.ann.score_candidates(toks, v) if v is not None else np.zeros(len(toks)))
        vl = build_user_embedding(recent_long, ctx.id_to_embedding, n_recent=N_RECENT_LONG)
        sem_long.extend(ctx.ann.score_candidates(toks, vl) if vl is not None else np.zeros(len(toks)))

    imp = np.repeat(np.arange(n), counts)
    cand = pd.DataFrame({"impression_id": imp, "article_id": arts,
                         "semantic_score": np.asarray(sem, float), "semantic_score_long": np.asarray(sem_long, float),
                         "bm25_score": np.asarray(bm, float)})
    if labeled:
        cand["clicked"] = np.asarray(labels, np.int64)
    if cand.empty:
        return cand, imp_ids, counts

    code = cand["article_id"].map(ctx.art_code)
    known = code.notna().to_numpy()
    code = code.fillna(-1).astype(np.int64).to_numpy()
    safe = np.where(known, code, 0)
    ts = secs[imp]
    hour = ts // 3600

    # --- user history: length + category / subcategory shares -------------
    h_imp, h_code, h_in10 = (np.asarray(x, np.int64) for x in (hist_imp, hist_code, hist_in10))
    h_in10 = h_in10.astype(bool)
    n50 = np.bincount(h_imp, minlength=n).astype(float)
    n10 = np.bincount(h_imp[h_in10], minlength=n).astype(float)
    cand_cat = np.where(known, ctx.cat_of_code[safe], ctx.n_cats - 1)
    cand_sub = np.where(known, ctx.sub_of_code[safe], ctx.n_subs - 1)
    h_cat, h_sub = ctx.cat_of_code[h_code], ctx.sub_of_code[h_code]
    cat50 = _share_lookup(h_imp * ctx.n_cats + h_cat, imp * ctx.n_cats + cand_cat)
    cat10 = _share_lookup(h_imp[h_in10] * ctx.n_cats + h_cat[h_in10], imp * ctx.n_cats + cand_cat)
    sub50 = _share_lookup(h_imp * ctx.n_subs + h_sub, imp * ctx.n_subs + cand_sub)
    with np.errstate(invalid="ignore", divide="ignore"):
        cand["cat_frac_50"] = np.where(n50[imp] > 0, cat50 / n50[imp], np.nan)
        cand["cat_frac_10"] = np.where(n10[imp] > 0, cat10 / n10[imp], np.nan)
        cand["subcat_frac_50"] = np.where(n50[imp] > 0, sub50 / n50[imp], np.nan)
    cand["n_hist"] = n_hist[imp]

    # --- article priors (earlier labelled files only) ----------------------
    p_impr = np.where(known, ctx.prior_impr[safe], 0).astype(float)
    p_clicks = np.where(known, ctx.prior_clicks[safe], 0).astype(float)
    cand["prior_impressions"] = p_impr
    cand["prior_clicks"] = p_clicks
    with np.errstate(invalid="ignore", divide="ignore"):
        cand["prior_ctr"] = np.where(p_impr > 0, p_clicks / p_impr, np.nan)

    # --- in-view popularity: exposures in strictly earlier full hours ------
    for w in INVIEW_WINDOWS_HOURS:
        cand[f"inview_prev_{w}h"] = np.where(known, ctx.inview.window(safe, hour, w), np.nan)
    fs = np.where(known, ctx.first_seen[safe], np.nan)
    cand["hours_since_first_seen"] = np.clip((ts - fs) / 3600.0, 0, None)

    cand = add_impression_relative_features(cand, V2_RELATIVE_SOURCE_COLS)

    # --- v1 features, exactly as generate_mind_reranked_submission.py ------
    rank = cand.groupby("impression_id")["semantic_score"].rank(ascending=False, method="first")
    cand["position"] = (rank - 1).astype(int)
    cand["position_ctr_prior"] = cand["position"].map(ctx.pos_ctr_prior).astype(float)
    cand["n_clicks_before"] = np.minimum(n_hist[imp], N_RECENT)
    cand["recency_weighted_click_count"] = cand["n_clicks_before"]
    cand["is_cold_start"] = (cand["n_clicks_before"] <= COLD_START_MAX_CLICKS).astype(int)
    cand["popularity_prior_ctr"] = cand["prior_ctr"].fillna(0.0)
    cand["freshness_hours"] = np.nan
    cand["category_match"] = (cat10 > 0).astype(int)
    cand["category_match_frac"] = cand["cat_frac_10"]
    cand["session_click_count_before"] = 0
    cand["session_impressions_before"] = 0
    cand["avg_dwell_time_before"] = np.nan

    float_cols = [c for c in set(V2_FEATURE_COLS) | set(FEATURE_COLS) if cand[c].dtype == np.float64]
    cand[float_cols] = cand[float_cols].astype(np.float32)
    return cand, imp_ids, counts
