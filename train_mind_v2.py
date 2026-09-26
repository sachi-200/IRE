#!/usr/bin/env python
"""Train the MIND-large v2 reranker on labelled MINDlarge_dev, with features
computed by src/mind_large_v2.py -- the SAME code
generate_mind_v2_submission.py runs on the unlabelled test set.

Leakage boundary: article click priors come from MINDlarge_train only
(dev's own labels are never features); in-view exposure counts use train +
dev behaviors (label-free), always strictly earlier full hours.

Sampled dev impressions are split 75% fit / 10% early-stopping / 15%
holdout. On the holdout it reports, side by side:
  semantic      -- Stage-1 semantic score alone
  v1_submitted  -- results/reranker_mind_model.txt (your current Codabench
                   model) on its own 14 features, computed exactly as
                   generate_mind_reranked_submission.py computes them
  v2            -- this model
with a paired bootstrap 95% CI on (v2 - v1). Only submit v2 if it wins.

Usage:
    python train_mind_v2.py --train-dir data/raw/mind_large/train \
        --dev-dir data/raw/mind_large/dev --test-dir data/raw/mind_large/test
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.bootstrap import bootstrap_ci
from src.config import ROOT
from src.mind_large_v2 import (BEH_COLS, BEH_DTYPES, CHUNK_SIZE, V2_FEATURE_COLS,
                               build_context, chunk_candidates)
from src.reranker import FEATURE_COLS, IMPROVED_LGBM_PARAMS, evaluate_scores, train_ranker

RESULTS_DIR = ROOT / "results"
METRICS = ["auc", "mrr", "ndcg5", "ndcg10"]


def build_training_rows(dev_path: Path, ctx, max_impressions: int, seed: int) -> pd.DataFrame:
    dev = pd.read_csv(dev_path, sep="\t", header=None, names=BEH_COLS, dtype=BEH_DTYPES)
    dev = dev.dropna(subset=["impressions"])
    if max_impressions and len(dev) > max_impressions:
        dev = dev.sample(n=max_impressions, random_state=seed)
    dev = dev.reset_index(drop=True)
    print(f"  building features for {len(dev):,} dev impressions...")

    keep = list(dict.fromkeys(V2_FEATURE_COLS + FEATURE_COLS + ["impression_id", "clicked"]))
    parts, t0 = [], time.time()
    for start in range(0, len(dev), CHUNK_SIZE):
        cand, _, _ = chunk_candidates(dev.iloc[start:start + CHUNK_SIZE], ctx, labeled=True)
        cand["impression_id"] += start  # chunk-local row index -> global
        parts.append(cand[keep])
        done = min(start + CHUNK_SIZE, len(dev))
        print(f"    {done:,}/{len(dev):,} impressions ({done / (time.time() - t0):.0f}/sec)")
    return pd.concat(parts, ignore_index=True)


def report(name, metrics_df):
    out = {}
    for m in METRICS:
        mean, lo, hi = bootstrap_ci(metrics_df[m].dropna().values)
        out[m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    print(f"  {name:<13} " + "  ".join(f"{m}={out[m]['mean']:.4f}" for m in METRICS))
    return out


def paired(base_df, new_df, label):
    out = {}
    for m in METRICS:
        merged = base_df[["impression_id", m]].merge(new_df[["impression_id", m]], on="impression_id",
                                                      suffixes=("_b", "_n")).dropna()
        mean, lo, hi = bootstrap_ci((merged[f"{m}_n"] - merged[f"{m}_b"]).values)
        sig = bool(lo > 0 or hi < 0)
        out[m] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "significant": sig}
        print(f"    {label} {m:<7} delta = {mean:+.4f}  (95% CI: [{lo:+.4f}, {hi:+.4f}])"
              f"  [{'SIGNIFICANT' if sig else 'not significant'}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--test-dir", required=True, help="only its news.tsv is read (keeps the BM25 corpus "
                                                      "identical to the submission's)")
    ap.add_argument("--max-impressions", type=int, default=100_000,
                    help="dev impressions to sample (~37 candidate rows each); lower it if memory is tight")
    ap.add_argument("--embeddings-name", default="mind_large",
                    help="data/features/embeddings_<name>.npz (reuses the v1 submission's cache)")
    ap.add_argument("--v1-model", default=str(RESULTS_DIR / "reranker_mind_model.txt"))
    ap.add_argument("--out-prefix", default=str(RESULTS_DIR / "mind_large_v2"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_dir, dev_dir, test_dir = Path(args.train_dir), Path(args.dev_dir), Path(args.test_dir)
    print("== context: corpus, indexes, priors (train labels only), in-view counts (train+dev) ==")
    ctx = build_context([train_dir, dev_dir, test_dir],
                        prior_files=[train_dir / "behaviors.tsv"],
                        inview_files=[train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv"],
                        embeddings_name=args.embeddings_name)

    print("== features ==")
    rows = build_training_rows(dev_dir / "behaviors.tsv", ctx, args.max_impressions, args.seed)
    del ctx

    ids = rows["impression_id"].unique()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_fit, n_es = int(0.75 * len(ids)), int(0.10 * len(ids))
    fit_ids, es_ids = set(ids[:n_fit]), set(ids[n_fit:n_fit + n_es])
    part = np.where(rows["impression_id"].isin(fit_ids), "fit",
                    np.where(rows["impression_id"].isin(es_ids), "es", "holdout"))
    print(f"  {len(rows):,} candidate rows; impressions: {n_fit:,} fit / {n_es:,} early-stop / "
          f"{len(ids) - n_fit - n_es:,} holdout")

    print("== training v2 (LambdaMART, early stopping on the 10% slice) ==")
    ranker = train_ranker(rows[part == "fit"], V2_FEATURE_COLS, valid_df=rows[part == "es"],
                          **IMPROVED_LGBM_PARAMS)
    print(f"  best iteration: {ranker.best_iteration_}")
    model_path = f"{args.out_prefix}_model.txt"
    ranker.booster_.save_model(model_path)
    gain = pd.Series(ranker.booster_.feature_importance("gain"), index=V2_FEATURE_COLS)
    print("  top-10 features by gain: " + ", ".join(gain.sort_values(ascending=False).index[:10]))

    print("== holdout comparison ==")
    hold = rows[part == "holdout"].copy()
    del rows
    hold["score_v2"] = ranker.predict(hold[V2_FEATURE_COLS])
    v1 = lgb.Booster(model_file=args.v1_model)
    hold["score_v1"] = v1.predict(hold[FEATURE_COLS].astype(float))
    per = {"semantic": evaluate_scores(hold, "semantic_score"),
           "v1_submitted": evaluate_scores(hold, "score_v1"),
           "v2": evaluate_scores(hold, "score_v2")}
    summary = {name: report(name, df) for name, df in per.items()}
    print("  paired bootstrap:")
    summary["delta_v2_vs_v1"] = paired(per["v1_submitted"], per["v2"], "v2 - v1")
    summary["n_holdout_impressions"] = int(len(per["v2"]))
    summary["best_iteration"] = int(ranker.best_iteration_)
    summary["feature_importance_gain"] = (gain / gain.sum()).round(4).sort_values(ascending=False).to_dict()

    with open(f"{args.out_prefix}_features.json", "w") as f:
        json.dump({"feature_cols": V2_FEATURE_COLS}, f, indent=2)
    with open(f"{args.out_prefix}_holdout.json", "w") as f:
        json.dump(summary, f, indent=2)
    auc_delta = summary["delta_v2_vs_v1"]["auc"]
    verdict = ("v2 beats the submitted model on holdout AUC -- worth submitting"
               if auc_delta["significant"] and auc_delta["mean"] > 0
               else "v2 does NOT significantly beat the submitted model on AUC -- don't submit")
    print(f"\n  {verdict}")
    print(f"  saved {model_path}, {args.out_prefix}_features.json, {args.out_prefix}_holdout.json")


if __name__ == "__main__":
    main()
