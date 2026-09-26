"""Assignment 2, Part I Q2: two-stage retrieve-then-rank pipeline.

Stage 1 (retrieval, reusing Assignment 1's candidate generators): score
each impression's own candidate set with BM25 and semantic similarity --
the same per-impression scoring run_eval_harness.py already does for its
"before" baseline -- using Part I.1's point-in-time recent_article_ids as
the query/user-vector source (not A1's coarser global click history), so
Stage 1 respects the same behavioural-window boundary as Stage 2's
features. Keep the top-K by the chosen PRIMARY method's score (K~100-200
per spec; a no-op for most impressions here since MIND/EB-NeRD's official
candidate lists are usually much smaller, but real for the long tail --
some MIND impressions carry 400+ candidates).

Stage 2 (re-rank): a LightGBM LambdaMART ranker trained on Part I.1's
engineered behavioural features (data/features/behavioral_features.parquet)
plus the two retrieval scores as extra features, one group per impression.
"""
import numpy as np
import pandas as pd

from src.bm25_index import BM25Index
from src.ann_index import ANNIndex
from src.embeddings import load_embeddings, embeddings_exist, compute_embeddings, save_embeddings
from src.query_builder import build_query, build_user_embedding

FEATURE_COLS = [
    "bm25_score", "semantic_score",
    "n_clicks_before", "recency_weighted_click_count", "is_cold_start",
    "popularity_prior_ctr", "freshness_hours",
    "category_match", "category_match_frac",
    "session_click_count_before", "session_impressions_before", "avg_dwell_time_before",
    "position_ctr_prior", "position",
]
# FEATURE_COLS above is the original Q2 feature set. It is frozen: the
# Codabench submission scripts and run_serving_benchmark.py build exactly
# these columns for results/reranker_<dataset>_model.txt.

# `position` is where the candidate sat in the dataset's own logged in-view
# list -- decided by the production system that collected the log, so a
# fresh request can't have it before ranking (Q9: "features unavailable at
# serving time"). Excluded from the improved set, kept only for the ablation.
SERVING_UNAVAILABLE_COLS = ["position", "position_ctr_prior"]

# per-candidate scores re-expressed RELATIVE to the rest of the impression's
# candidates -- LambdaMART only ever compares candidates within one group,
# so "best semantic match in this impression" is more useful than a raw
# cosine whose scale drifts from user to user
RELATIVE_SOURCE_COLS = ["semantic_score", "bm25_score", "popularity_prior_ctr",
                        "hours_since_first_seen", "cum_impressions_prior",
                        "impressions_prior_1h", "ctr_prior_24h"]
RELATIVE_FEATURE_COLS = ["n_candidates"] + [
    f"{c}_{kind}" for c in RELATIVE_SOURCE_COLS for kind in ("pct_rank", "z", "gap")]

IMPROVED_FEATURE_COLS = (
    [c for c in FEATURE_COLS if c not in SERVING_UNAVAILABLE_COLS]
    + ["days_since_last_click", "hours_since_first_seen", "cum_impressions_prior", "cum_clicks_prior"]
    + [f"{kind}_prior_{h}h" for h in (1, 24) for kind in ("clicks", "impressions", "ctr")]
    + RELATIVE_FEATURE_COLS
)

# early-stopped LambdaMART for the improved set (the base set keeps the
# original fixed 200-tree config so its numbers stay reproducible)
IMPROVED_LGBM_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                            min_child_samples=50, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=1.0)


def add_impression_relative_features(df: pd.DataFrame, source_cols=RELATIVE_SOURCE_COLS) -> pd.DataFrame:
    """Adds RELATIVE_FEATURE_COLS: for each source column, its percentile
    rank within the impression (1.0 = highest), its z-score within the
    impression, and its gap to the impression's maximum; plus the
    impression's candidate count. Only uses the candidate set itself (known
    before ranking), never labels. Call AFTER cap_top_k so the reranker sees
    the same candidate set these were computed over."""
    out = df.copy()
    g = out.groupby("impression_id", observed=True)
    out["n_candidates"] = g["article_id"].transform("size").astype(float)
    for c in source_cols:
        col = out[c].astype(float)
        gc_ = col.groupby(out["impression_id"], observed=True)
        mean, std, mx = gc_.transform("mean"), gc_.transform("std"), gc_.transform("max")
        out[f"{c}_pct_rank"] = gc_.rank(ascending=False, pct=True, method="average")
        out[f"{c}_z"] = ((col - mean) / std.replace(0, np.nan)).fillna(0.0).where(col.notna())
        out[f"{c}_gap"] = col - mx
    return out


def build_bm25_index(articles: pd.DataFrame, dataset: str):
    sub = articles[articles["dataset"] == dataset].copy()
    sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
    index = BM25Index().fit(sub["article_id"].tolist(), sub["text"].tolist())
    return index, dict(zip(sub["article_id"], sub["text"]))


def build_semantic_index(articles: pd.DataFrame, dataset: str):
    if embeddings_exist(dataset):
        article_ids, embeddings = load_embeddings(dataset)
    else:
        sub = articles[articles["dataset"] == dataset].copy()
        sub["text"] = (sub["title"].fillna("") + " " + sub["abstract"].fillna("")).str.strip()
        article_ids, embeddings = compute_embeddings(sub["article_id"].tolist(), sub["text"].tolist())
        save_embeddings(dataset, article_ids, embeddings)
    index = ANNIndex().fit(article_ids, embeddings)
    return index, dict(zip(article_ids, embeddings))


def add_retrieval_scores(candidates_df: pd.DataFrame, click_hist: pd.DataFrame,
                          bm25_index: BM25Index, bm25_lookup: dict,
                          semantic_index: ANNIndex, id_to_embedding: dict) -> pd.DataFrame:
    """candidates_df: one row per (impression_id, article_id) candidate
    (a slice of behavioral_features.parquet). Adds bm25_score/semantic_score,
    computed once per impression (all its candidates share one query) via
    score_docs() -- vectorized over that impression's candidate list rather
    than one dot product per row.
    """
    hist_lookup = click_hist.set_index("impression_id")["recent_article_ids"]

    out = candidates_df.copy()
    out["bm25_score"] = 0.0
    out["semantic_score"] = 0.0

    for iid, grp in out.groupby("impression_id", observed=True):
        recent_ids = hist_lookup.get(iid, [])
        candidates = grp["article_id"].tolist()

        query = build_query(recent_ids, bm25_lookup)
        bm25_scores = bm25_index.score_docs(candidates, query) if query else {c: 0.0 for c in candidates}

        user_vec = build_user_embedding(recent_ids, id_to_embedding)
        sem_scores = (semantic_index.score_docs(candidates, user_vec) if user_vec is not None
                      else {c: 0.0 for c in candidates})

        out.loc[grp.index, "bm25_score"] = [bm25_scores.get(c, 0.0) for c in candidates]
        out.loc[grp.index, "semantic_score"] = [sem_scores.get(c, 0.0) for c in candidates]
    return out


def cap_top_k(scored_df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Stage 1's 'retrieve top-K': keep each impression's K highest-scoring
    candidates by score_col. A no-op for impressions with <= K candidates
    already (the common case here); for larger ones, an impression whose
    true click falls outside the top-K becomes an honest retrieval miss --
    evaluate_scores() skips impressions with no positive left, exactly like
    run_eval_harness.py already does for degenerate cases.
    """
    ranked = scored_df.sort_values(["impression_id", score_col], ascending=[True, False])
    rank_in_impression = ranked.groupby("impression_id", observed=True).cumcount()
    return ranked[rank_in_impression < k]


def _prepare_X(df: pd.DataFrame, feature_cols=FEATURE_COLS) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for col in ("is_cold_start", "category_match"):
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X  # LightGBM handles remaining NaNs (e.g. MIND's freshness_hours) natively


def _grouped(df: pd.DataFrame, feature_cols):
    df = df.sort_values("impression_id")  # LightGBM needs each group's rows contiguous
    groups = df.groupby("impression_id", observed=True, sort=False).size().to_numpy()
    return _prepare_X(df, feature_cols), df["clicked"].to_numpy(), groups


def train_ranker(train_df: pd.DataFrame, feature_cols=FEATURE_COLS, valid_df: pd.DataFrame = None,
                 early_stopping_rounds: int = 100, **lgbm_kwargs):
    """valid_df (optional): a held-out slice of TRAIN impressions used only
    to pick the number of trees (early stopping on nDCG@10) -- never the
    val/test splits that metrics are reported on."""
    import lightgbm as lgb
    X, y, groups = _grouped(train_df, feature_cols)

    params = dict(objective="lambdarank", metric="ndcg", n_estimators=200,
                  learning_rate=0.05, num_leaves=31, min_child_samples=20, verbosity=-1)
    params.update(lgbm_kwargs)
    ranker = lgb.LGBMRanker(**params)
    if valid_df is None:
        ranker.fit(X, y, group=groups)
    else:
        Xv, yv, gv = _grouped(valid_df, feature_cols)
        ranker.fit(X, y, group=groups, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                   callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    return ranker


def score_ranker(ranker, df: pd.DataFrame, feature_cols=FEATURE_COLS) -> np.ndarray:
    return ranker.predict(_prepare_X(df, feature_cols))


def evaluate_scores(df: pd.DataFrame, score_col: str, label_col: str = "clicked") -> pd.DataFrame:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 for one score column -- mirrors
    run_eval_harness.py's per-impression loop. Impressions with <2
    candidates or no positive label are skipped (metric undefined)."""
    from src.ranking_metrics import auc, mrr, ndcg_at_k
    rows = []
    for iid, grp in df.groupby("impression_id", observed=True):
        scores = grp[score_col].to_numpy()
        labels = grp[label_col].to_numpy()
        if len(scores) < 2 or labels.sum() == 0:
            continue
        rows.append({
            "impression_id": iid,
            "auc": auc(scores, labels), "mrr": mrr(scores, labels),
            "ndcg5": ndcg_at_k(scores, labels, 5), "ndcg10": ndcg_at_k(scores, labels, 10),
        })
    return pd.DataFrame(rows)
