#!/usr/bin/env python
"""Q2: two-stage retrieve-then-rank pipeline.

1. Retrieval: score each impression's official candidate set with BM25 +
   semantic similarity (Assignment 1's candidate generators, queried from
   Assignment 2 Q1's point-in-time click history) and keep the top-K.
2. Re-rank: train a LightGBM LambdaMART ranker over Q1's engineered
   behavioural features plus the two retrieval scores.
3. Report AUC/MRR/nDCG@5/nDCG@10 before (Stage 1's own score) and after
   (the trained reranker's score), with bootstrap 95% CI, on val and test.

Requires data/features/behavioral_features.parquet and
data/features/click_history_features.parquet (Assignment 2 Q1) --
run build_behavioral_features.py first.

--feature-set improved (default) additionally trains the original feature
set on the same sample and reports a paired-bootstrap comparison, plus the
Q9 with/without-position-features ablation (see run_improved).

Usage:
    python run_reranker.py --dataset mind
    python run_reranker.py --dataset all --k 150 --primary-method semantic
    python run_reranker.py --dataset mind --feature-set base   # original Q2 run
"""
import argparse
import gc
import json
import random
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, FEATURES_DIR, ROOT
from src.reranker import (
    build_bm25_index, build_semantic_index, add_retrieval_scores, cap_top_k,
    train_ranker, score_ranker, evaluate_scores, add_impression_relative_features,
    FEATURE_COLS, IMPROVED_FEATURE_COLS, SERVING_UNAVAILABLE_COLS, IMPROVED_LGBM_PARAMS,
)
from src.bootstrap import bootstrap_ci

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sample_impression_ids(ids, max_impressions, seed: int = 42):
    ids = list(ids)
    if max_impressions and len(ids) > max_impressions:
        random.seed(seed)
        ids = random.sample(ids, max_impressions)
    return ids


def load_sampled_candidates(dataset: str, max_train_impressions: int, max_eval_impressions: int):
    """Picks which impressions to use BEFORE reading their features, then
    reads behavioral_features.parquet / click_history_features.parquet with
    a pyarrow filter on that exact impression_id list.

    behavioral_features.parquet holds every candidate row for BOTH datasets
    (9M+ total; MIND alone is ~8.6M) -- reading it in full and sampling down
    to a few thousand impressions afterwards, as an earlier version of this
    script did, means paying MIND's full memory cost before ever discarding
    anything. A lightweight 2-column projection (just impression_id + split)
    is cheap enough to read in full, so sampling can happen on that instead,
    and the real (23-column) read only ever touches the sampled rows.
    """
    proj = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet",
                            columns=["impression_id", "split"],
                            filters=[("dataset", "==", dataset)])
    proj = proj.drop_duplicates("impression_id")

    sampled_ids = []
    for split_name, n in [("train", max_train_impressions),
                           ("val", max_eval_impressions), ("test", max_eval_impressions)]:
        ids = proj.loc[proj["split"] == split_name, "impression_id"].tolist()
        sampled_ids.extend(sample_impression_ids(ids, n))
    del proj

    row_filter = [("dataset", "==", dataset), ("impression_id", "in", sampled_ids)]
    behavioral = pd.read_parquet(FEATURES_DIR / "behavioral_features.parquet", filters=row_filter)
    click_hist = pd.read_parquet(FEATURES_DIR / "click_history_features.parquet", filters=row_filter)
    return behavioral, click_hist


def report_metric(name, series):
    vals = series.dropna().values
    mean, lo, hi = bootstrap_ci(vals)
    if mean is None:
        print(f"    {name:<8} = n/a (no valid data)")
        return None
    print(f"    {name:<8} = {mean:.4f}  (95% CI: [{lo:.4f}, {hi:.4f}], n={len(vals)})")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(vals)}


def paired_delta(base_metrics: pd.DataFrame, new_metrics: pd.DataFrame, metric: str, label: str):
    """Paired bootstrap 95% CI over per-impression (new - base) deltas; a
    gain is only called significant when the CI excludes zero."""
    merged = base_metrics[["impression_id", metric]].merge(
        new_metrics[["impression_id", metric]], on="impression_id", suffixes=("_base", "_new")).dropna()
    diff = (merged[f"{metric}_new"] - merged[f"{metric}_base"]).values
    mean, lo, hi = bootstrap_ci(diff)
    if mean is None:
        return None
    significant = bool(lo > 0 or hi < 0)
    print(f"    {label:<28} {metric:<7} delta = {mean:+.4f}  (95% CI: [{lo:+.4f}, {hi:+.4f}])"
          f"  [{'SIGNIFICANT' if significant else 'not significant'}]")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(diff), "significant": significant}


def run_improved(dataset: str, capped: pd.DataFrame, primary_col: str, primary_method: str, seed: int = 42):
    """Improved Stage 2 (see src/reranker.py IMPROVED_FEATURE_COLS):
    within-impression relative features + extra leak-free popularity/
    freshness signals, early-stopped on a held-out 10% of TRAIN impressions.
    Trains three models on the same data so the report is a clean ablation:
      base               -- original Q2 features/config (never saved here, so
                            results/reranker_<ds>_model.txt -- which the
                            Codabench scripts load -- is left untouched)
      improved           -- serving-safe improved set (saved as *_improved_*)
      improved+position  -- improved + logged-position features (Q9: metrics
                            WITH features unavailable at serving time)
    """
    capped = add_impression_relative_features(capped)
    train_df = capped[capped["split"] == "train"]
    eval_splits = [(s, capped[capped["split"] == s]) for s in ("val", "test")]
    del capped
    gc.collect()

    train_ids = train_df["impression_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    es_ids = set(rng.choice(train_ids, size=max(1, len(train_ids) // 10), replace=False))
    is_es = train_df["impression_id"].astype(str).isin(es_ids).to_numpy()
    fit_df, es_df = train_df[~is_es], train_df[is_es]

    print(f"  Stage 2: {train_df['impression_id'].nunique()} train impressions "
          f"({fit_df['impression_id'].nunique()} fit / {es_df['impression_id'].nunique()} early-stopping holdout)")
    models = {}
    print("    training base (original Q2 features)...")
    models["base"] = (train_ranker(train_df, FEATURE_COLS), FEATURE_COLS)
    print("    training improved (serving-safe)...")
    models["improved"] = (train_ranker(fit_df, IMPROVED_FEATURE_COLS, valid_df=es_df, **IMPROVED_LGBM_PARAMS),
                          IMPROVED_FEATURE_COLS)
    with_pos = IMPROVED_FEATURE_COLS + SERVING_UNAVAILABLE_COLS
    print("    training improved+position (Q9 ablation)...")
    models["improved+position"] = (train_ranker(fit_df, with_pos, valid_df=es_df, **IMPROVED_LGBM_PARAMS),
                                   with_pos)
    del train_df, fit_df, es_df
    gc.collect()

    improved = models["improved"][0]
    improved.booster_.save_model(str(RESULTS_DIR / f"reranker_{dataset}_improved_model.txt"))
    print(f"    improved model: best iteration = {improved.best_iteration_}")
    gain = pd.Series(improved.booster_.feature_importance("gain"), index=IMPROVED_FEATURE_COLS)
    print("    top-10 features by gain: " + ", ".join(gain.sort_values(ascending=False).index[:10]))

    report = {"feature_importance_gain": (gain / gain.sum()).round(4).sort_values(ascending=False).to_dict()}
    for split_name, split_df in eval_splits:
        if split_df.empty:
            continue
        split_df = split_df.copy()
        print(f"\n  -- {split_name} ({split_df['impression_id'].nunique()} impressions) --")
        per_model = {"before": evaluate_scores(split_df, primary_col)}
        for name, (model, cols) in models.items():
            split_df[f"score_{name}"] = score_ranker(model, split_df, cols)
            per_model[name] = evaluate_scores(split_df, f"score_{name}")

        split_report = {"n_impressions": len(per_model["before"])}
        for name, metrics_df in per_model.items():
            label = f"BEFORE ({primary_method} score alone)" if name == "before" else f"AFTER [{name}]"
            print(f"  {label}:")
            split_report[name] = {m: report_metric(m, metrics_df[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        print("  PAIRED bootstrap 95% CI:")
        split_report["delta_improved_vs_base"] = {
            m: paired_delta(per_model["base"], per_model["improved"], m, "improved - base")
            for m in ["auc", "mrr", "ndcg5", "ndcg10"]}
        split_report["delta_position_features"] = {
            m: paired_delta(per_model["improved"], per_model["improved+position"], m, "(+position) - improved")
            for m in ["auc", "mrr", "ndcg5", "ndcg10"]}
        report[split_name] = split_report

    with open(RESULTS_DIR / f"reranker_{dataset}_improved_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/reranker_{dataset}_improved_summary.json, "
          f"results/reranker_{dataset}_improved_model.txt")
    return report


def run(dataset: str, k: int, primary_method: str, max_train_impressions: int, max_eval_impressions: int,
        feature_set: str = "improved"):
    print(f"\n=== Q2 reranker: {dataset} (K={k}, Stage-1 method={primary_method}, "
          f"feature set={feature_set}) ===")
    combined_raw, click_hist = load_sampled_candidates(dataset, max_train_impressions, max_eval_impressions)
    split_counts = combined_raw["split"].value_counts()
    print(f"  sampled {combined_raw['impression_id'].nunique()} impressions "
          f"({split_counts.get('train', 0)} train / {split_counts.get('val', 0)} val / "
          f"{split_counts.get('test', 0)} test candidate rows)")

    articles = pd.read_parquet(PROCESSED_DIR / "articles.parquet", filters=[("dataset", "==", dataset)])
    print("  building BM25 + semantic indexes...")
    bm25_index, bm25_lookup = build_bm25_index(articles, dataset)
    semantic_index, id_to_embedding = build_semantic_index(articles, dataset)

    print("  Stage 1: scoring each impression's candidates with both retrievers...")
    scored = add_retrieval_scores(combined_raw, click_hist, bm25_index, bm25_lookup,
                                   semantic_index, id_to_embedding)
    # only `scored` is needed from here on -- these were sizeable full copies
    # (one row per candidate) and Stage 2 (LightGBM) needs its own headroom
    del combined_raw, click_hist
    gc.collect()

    primary_col = "bm25_score" if primary_method == "bm25" else "semantic_score"
    capped = cap_top_k(scored, primary_col, k)
    print(f"  kept top-{k}: {len(scored)} -> {len(capped)} candidate rows")
    del scored
    gc.collect()

    if feature_set == "improved":
        return run_improved(dataset, capped, primary_col, primary_method)

    train_df = capped[capped["split"] == "train"]
    val_df = capped[capped["split"] == "val"]
    test_df = capped[capped["split"] == "test"]
    del capped
    gc.collect()

    print(f"  Stage 2: training LightGBM LambdaMART on {train_df['impression_id'].nunique()} "
          f"train impressions ({len(train_df)} rows)...")
    ranker = train_ranker(train_df)
    ranker.booster_.save_model(str(RESULTS_DIR / f"reranker_{dataset}_model.txt"))
    del train_df
    gc.collect()

    report = {}
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if split_df.empty:
            continue
        split_df = split_df.copy()
        split_df["reranker_score"] = score_ranker(ranker, split_df)

        print(f"\n  -- {split_name} ({split_df['impression_id'].nunique()} impressions) --")
        print(f"  BEFORE re-ranking ({primary_method} score alone):")
        before = evaluate_scores(split_df, primary_col)
        before_metrics = {m: report_metric(m, before[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        print("  AFTER re-ranking (LightGBM LambdaMART):")
        after = evaluate_scores(split_df, "reranker_score")
        after_metrics = {m: report_metric(m, after[m]) for m in ["auc", "mrr", "ndcg5", "ndcg10"]}

        report[split_name] = {"before": before_metrics, "after": after_metrics, "n_impressions": len(before)}

    with open(RESULTS_DIR / f"reranker_{dataset}_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/reranker_{dataset}_summary.json, "
          f"results/reranker_{dataset}_model.txt")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mind", "ebnerd", "all"], default="all")
    ap.add_argument("--k", type=int, default=150, help="top-K kept after Stage 1 retrieval (spec: 100-200)")
    ap.add_argument("--primary-method", choices=["bm25", "semantic"], default="semantic",
                     help="which Assignment 1 candidate generator is Stage 1 / the 'before' baseline")
    ap.add_argument("--max-train-impressions", type=int, default=20000)
    ap.add_argument("--max-eval-impressions", type=int, default=5000)
    ap.add_argument("--feature-set", choices=["improved", "base"], default="improved",
                     help="improved: new features + early stopping, saved as results/reranker_<ds>_improved_*. "
                          "base: the original Q2 run -- NOTE this overwrites results/reranker_<ds>_model.txt, "
                          "the model the Codabench submission scripts load")
    args = ap.parse_args()

    datasets = ["mind", "ebnerd"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run(ds, args.k, args.primary_method, args.max_train_impressions, args.max_eval_impressions,
            args.feature_set)


if __name__ == "__main__":
    main()
