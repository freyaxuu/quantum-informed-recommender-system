#!/usr/bin/env python3
"""Rebuild and verify the frozen Table 1 prior without repeating model selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from qiml_prior_experiments import ExperimentConfig, save_prior_spec
from svd_prior_search import (
    build_user_itemtype_prior,
    compute_item_clusters,
    compute_user_clusters,
    load_context,
    split_fingerprint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument(
        "--selection_manifest", type=Path,
        default=Path("results/static_v1_corrected/svd_prior_search_common_reference/"
                     "best_config/selected/selection_manifest.json"),
    )
    parser.add_argument("--out_dir", type=Path,
                        default=Path("results/static_v1_corrected/rebuilt_table1_prior"))
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection_manifest.read_text())
    fixed = selection["fixed_before_confirmation"]
    cfg = ExperimentConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, seeds=list(range(10)),
        epochs=selection["epochs"], factors=selection["factors"],
        batch_size=fixed["batch_size"], lr=selection["lr"], l2=selection["l2"],
        lambda_kl=selection["lambda_kl"], lambda_mmd=selection["lambda_mmd"],
        lambda_peak=0.0, soft_tau=selection["soft_tau"],
        residual_bins=fixed["residual_bins"], dot_bins=fixed["dot_bins"],
        user_clusters=selection["user_clusters"], item_clusters=selection["item_clusters"],
        top_k=fixed["top_k"], max_candidates=None, include_oracle=False,
    )
    if selection["candidate"] != "cond_user_itemtype" or (cfg.user_clusters, cfg.item_clusters, cfg.residual_bins) != (16, 16, 16):
        raise ValueError("Expected the frozen Table 1 cond_user_itemtype 16x16x16 configuration.")
    ctx = load_context(cfg)
    partitions = dict(zip(("train", "checkpoint_val", "selection_val", "confirmation"), ctx.all_splits))
    fingerprints = {name: split_fingerprint(df) for name, df in partitions.items()}
    if fingerprints != selection["split_fingerprints"]:
        raise ValueError("Dataset/split fingerprints differ from the original Table 1 run.")
    prior = build_user_itemtype_prior(
        ctx, compute_user_clusters(ctx, cfg.user_clusters, seed=42),
        compute_item_clusters(ctx, cfg.item_clusters, seed=42),
        cfg.user_clusters, cfg.item_clusters, selection["candidate"],
    )
    with np.load(selection["empirical_prior"], allow_pickle=False) as saved:
        for name, values in (("centers", prior.centers), ("probs", prior.probs)):
            if saved[name].shape != values.shape or not np.allclose(saved[name], values, rtol=0.0, atol=1e-12):
                raise ValueError(f"Rebuilt {name} differ from the frozen empirical prior.")
    report = {
        "verified": True,
        "selection_manifest": str(args.selection_manifest),
        "prior_shape": list(prior.probs.shape),
        "clustering_seed": 42,
        "rows": {name: len(df) for name, df in partitions.items()},
        "split_fingerprints": fingerprints,
        "input_sha256": {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                         for name in ("u.data", "u.item")},
        "saved_prior_matches_atol": 1e-12,
    }
    if not args.verify_only:
        if args.out_dir.exists() and any(args.out_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty output: {args.out_dir}")
        save_prior_spec(prior, args.out_dir)
        (args.out_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
