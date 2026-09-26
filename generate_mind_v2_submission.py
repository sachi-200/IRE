#!/usr/bin/env python
"""Codabench MIND submission with the v2 reranker (train_mind_v2.py).

Features come from src/mind_large_v2.py -- the same function the model was
trained with. For scoring, article click priors use train + dev labels
(both strictly before the test week) and in-view exposure counts use train
+ dev + test behaviors (label-free, strictly earlier full hours).

Output format is identical to generate_mind_reranked_submission.py:
"{impression_id} [rank_of_candidate_1,...]" in the candidates' original
order, streamed straight into the zip (no plaintext copy on disk -- the
disk-quota fix). That script and results/reranker_mind_model.txt are not
touched; this writes submission_v2.zip.

Usage:
    python generate_mind_v2_submission.py --train-dir data/raw/mind_large/train \
        --dev-dir data/raw/mind_large/dev --test-dir data/raw/mind_large/test
    # smoke test first (checks format on the first 2000 impressions):
    python generate_mind_v2_submission.py ... --limit 2000 --zip submission_v2_smoke.zip
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.config import ROOT
from src.mind_large_v2 import V2_FEATURE_COLS, build_context, chunk_candidates, iter_behaviors

RESULTS_DIR = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dev-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--model-prefix", default=str(RESULTS_DIR / "mind_large_v2"))
    ap.add_argument("--embeddings-name", default="mind_large")
    ap.add_argument("--out", default="prediction.txt")
    ap.add_argument("--zip", default="submission_v2.zip")
    ap.add_argument("--limit", type=int, default=None, help="only score the first N test impressions (smoke test)")
    args = ap.parse_args()

    with open(f"{args.model_prefix}_features.json") as f:
        feature_cols = json.load(f)["feature_cols"]
    assert feature_cols == V2_FEATURE_COLS, "model was trained with a different feature list -- retrain"
    booster = lgb.Booster(model_file=f"{args.model_prefix}_model.txt")

    train_dir, dev_dir, test_dir = Path(args.train_dir), Path(args.dev_dir), Path(args.test_dir)
    test_path = test_dir / "behaviors.tsv"
    print("== context: corpus, indexes, priors (train+dev labels), in-view counts (train+dev+test) ==")
    ctx = build_context([train_dir, dev_dir, test_dir],
                        prior_files=[train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv"],
                        inview_files=[train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv", test_path],
                        embeddings_name=args.embeddings_name)

    print(f"== scoring {test_path} ==")
    n_written, t0 = 0, time.time()
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf, \
            zf.open(Path(args.out).name, "w") as raw, \
            io.TextIOWrapper(raw, encoding="utf-8") as out_f:
        for chunk in iter_behaviors(test_path):
            if args.limit is not None:
                chunk = chunk.iloc[:max(args.limit - n_written, 0)]
                if chunk.empty:
                    break
            cand, imp_ids, counts = chunk_candidates(chunk, ctx, labeled=False)
            if len(cand):
                cand["score"] = booster.predict(cand[feature_cols])
                # rank 1 = best; ties keep original order (same as the v1 script's stable argsort)
                ranks = (cand.groupby("impression_id")["score"]
                         .rank(ascending=False, method="first").astype(int).to_numpy())
            else:
                ranks = np.array([], dtype=int)
            offsets = np.concatenate([[0], np.cumsum(counts)])
            for i, iid in enumerate(imp_ids):
                r = ranks[offsets[i]:offsets[i + 1]]
                out_f.write(f"{iid} [" + ",".join(map(str, r)) + "]\n")
            n_written += len(chunk)
            print(f"  {n_written:,} impressions ({n_written / (time.time() - t0):.0f}/sec, "
                  f"{(time.time() - t0) / 60:.1f} min)")

    print(f"wrote {n_written:,} lines into {args.zip} (arcname={Path(args.out).name}) -- ready to upload")


if __name__ == "__main__":
    main()
