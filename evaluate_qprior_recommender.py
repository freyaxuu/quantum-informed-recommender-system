#!/usr/bin/env python3
"""Evaluate whether a trained QCBM prior preserves recommender gains.

This is Stage 2 of the prior-first QIML workflow.  It keeps the winning prior
structure fixed, swaps the empirical conditional table for QCBM-learned tables,
and re-runs paired recommender experiments.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qiml_prior_experiments import (
    ExperimentConfig,
    MatrixFactorization,
    PriorSpec,
    add_indices,
    add_stat_columns,
    build_prior_specs,
    collect_all_metrics,
    compare_distributions,
    distribution_metrics,
    fit_stat_biases,
    load_movielens,
    normalize_probs,
    ranking_metrics,
    rating_metrics,
    subgroup_metrics,
    temporal_split,
    train_model,
)


def load_flat_prior_as_table(path: Path, shape: tuple[int, int]) -> np.ndarray:
    flat = np.load(path).astype(np.float64)
    expected = int(np.prod(shape))
    if flat.shape != (expected,):
        raise ValueError(f"{path} has shape {flat.shape}; expected flat vector of length {expected}.")
    return normalize_probs(flat.reshape(shape))


def load_empirical_prior(path: Path) -> Dict[str, np.ndarray | str]:
    data = np.load(path, allow_pickle=True)
    return {
        "name": str(data["name"]),
        "value_kind": str(data["value_kind"]),
        "mode": str(data["mode"]),
        "centers": data["centers"].astype(np.float64),
        "probs": data["probs"].astype(np.float64),
        "description": str(data["description"]) if "description" in data else "",
    }


def make_prior_like(base: PriorSpec, name: str, probs: np.ndarray, description: str) -> PriorSpec:
    return PriorSpec(
        name=name,
        value_kind=base.value_kind,
        centers=base.centers,
        probs=probs,
        mode=base.mode,
        group_keys=base.group_keys,
        description=description,
        group_col=base.group_col,
    )


def prepare_data_and_priors(args, cfg: ExperimentConfig):
    ratings, items = load_movielens(cfg.data_dir)
    train, val, test = temporal_split(ratings)
    train, val, test, user_to_idx, item_to_idx = add_indices(train, val, test)
    mu = float(train["rating"].mean())
    stat_user_bias, stat_item_bias = fit_stat_biases(train, mu)
    train = add_stat_columns(train, stat_user_bias, stat_item_bias, mu)
    val = add_stat_columns(val, stat_user_bias, stat_item_bias, mu)
    test = add_stat_columns(test, stat_user_bias, stat_item_bias, mu)

    # Rebuild the grouping columns exactly as Stage 1 did.
    specs = build_prior_specs(cfg, train, val, test, items, user_to_idx, item_to_idx)
    base = next((spec for spec in specs if spec.name == args.candidate), None)
    if base is None:
        names = ", ".join(spec.name for spec in specs)
        raise ValueError(f"Could not rebuild candidate {args.candidate}. Available: {names}")

    saved = load_empirical_prior(args.empirical_prior)
    if tuple(saved["probs"].shape) != tuple(base.probs.shape):
        raise ValueError(
            f"Saved empirical prior shape {saved['probs'].shape} does not match rebuilt "
            f"{args.candidate} shape {base.probs.shape}."
        )

    empirical = make_prior_like(base, args.candidate, saved["probs"], "Empirical Stage 1 winning prior")
    qcbm = make_prior_like(
        base,
        f"{args.candidate}_qcbm",
        load_flat_prior_as_table(args.qcbm_prior, empirical.probs.shape),
        "QCBM-learned prior reshaped and row-normalized to conditional table",
    )
    finite = make_prior_like(
        base,
        f"{args.candidate}_qcbm_finite_shots",
        load_flat_prior_as_table(args.finite_shot_prior, empirical.probs.shape),
        "Finite-shot QCBM prior reshaped and row-normalized to conditional table",
    )

    return train, val, test, user_to_idx, item_to_idx, mu, empirical, qcbm, finite


def collect_stage2_metrics(
    variant: str,
    seed: int,
    model: MatrixFactorization,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    train_prior: Optional[PriorSpec],
    empirical_reference: PriorSpec,
    cfg: ExperimentConfig,
    n_items: int,
):
    row = collect_all_metrics(
        variant=variant,
        candidate=empirical_reference.name,
        seed=seed,
        model=model,
        train=train,
        val=val,
        test=test,
        prior=train_prior,
        cfg=cfg,
        n_items=n_items,
    )
    pred, dot = model.predict(
        test["user_idx"].to_numpy(np.int64),
        test["item_idx"].to_numpy(np.int64),
        return_dot=True,
    )
    if empirical_reference.value_kind == "rating":
        values = pred
    elif empirical_reference.value_kind == "residual":
        values = pred - model.mu - test["bu_stat"].to_numpy() - test["bi_stat"].to_numpy()
    else:
        values = dot
    empirical_dist = distribution_metrics(values, empirical_reference, test, cfg)
    row.update({f"empirical_dist_{k}": v for k, v in empirical_dist.items()})
    if train_prior is None:
        row["train_prior_name"] = "none"
    else:
        row["train_prior_name"] = train_prior.name
    return row


def summarize_stage2(out_dir: Path):
    metrics = pd.read_csv(out_dir / "metrics.csv")
    base = metrics[metrics["variant"] == "baseline"][
        ["seed", "test_rmse", "empirical_dist_kl", "empirical_dist_js", "empirical_dist_mmd_l2"]
    ].rename(
        columns={
            "test_rmse": "baseline_test_rmse",
            "empirical_dist_kl": "baseline_empirical_dist_kl",
            "empirical_dist_js": "baseline_empirical_dist_js",
            "empirical_dist_mmd_l2": "baseline_empirical_dist_mmd_l2",
        }
    )
    merged = metrics.merge(base, on="seed", how="left")
    merged["rmse_gain_vs_baseline"] = merged["baseline_test_rmse"] - merged["test_rmse"]
    merged["dist_kl_gain_vs_baseline"] = merged["baseline_empirical_dist_kl"] - merged["empirical_dist_kl"]
    merged["dist_js_gain_vs_baseline"] = merged["baseline_empirical_dist_js"] - merged["empirical_dist_js"]
    merged["dist_mmd_gain_vs_baseline"] = merged["baseline_empirical_dist_mmd_l2"] - merged["empirical_dist_mmd_l2"]
    merged.to_csv(out_dir / "metrics_with_gains.csv", index=False)

    summary = (
        merged.groupby("variant", dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_rmse_gain=("rmse_gain_vs_baseline", "mean"),
            mean_test_mae=("test_mae", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_ndcg_at_k=("ndcg_at_k", "mean"),
            mean_map_at_k=("map_at_k", "mean"),
            mean_empirical_dist_kl=("empirical_dist_kl", "mean"),
            mean_empirical_dist_js=("empirical_dist_js", "mean"),
            mean_empirical_dist_mmd=("empirical_dist_mmd_l2", "mean"),
            mean_dist_kl_gain=("dist_kl_gain_vs_baseline", "mean"),
            mean_dist_js_gain=("dist_js_gain_vs_baseline", "mean"),
            mean_dist_mmd_gain=("dist_mmd_gain_vs_baseline", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    empirical = summary[summary["variant"] == "empirical_prior"].iloc[0]
    rows = []
    for variant in ["qcbm_prior", "finite_shot_qcbm_prior", "shuffled_prior", "uniform_prior", "random_prior"]:
        if variant not in set(summary["variant"]):
            continue
        row = summary[summary["variant"] == variant].iloc[0].to_dict()
        rmse_den = float(empirical["mean_rmse_gain"])
        dist_den = float(empirical["mean_dist_kl_gain"])
        row["rmse_preservation_vs_empirical"] = (
            float(row["mean_rmse_gain"]) / rmse_den if abs(rmse_den) > 1e-12 else np.nan
        )
        row["dist_kl_preservation_vs_empirical"] = (
            float(row["mean_dist_kl_gain"]) / dist_den if abs(dist_den) > 1e-12 else np.nan
        )
        row["beats_shuffled_rmse"] = np.nan
        row["beats_shuffled_dist_kl"] = np.nan
        rows.append(row)

    preservation = pd.DataFrame(rows)
    if not preservation.empty and "shuffled_prior" in set(summary["variant"]):
        shuffled = summary[summary["variant"] == "shuffled_prior"].iloc[0]
        preservation["beats_shuffled_rmse"] = preservation["mean_rmse_gain"] > float(shuffled["mean_rmse_gain"])
        preservation["beats_shuffled_dist_kl"] = preservation["mean_empirical_dist_kl"] < float(
            shuffled["mean_empirical_dist_kl"]
        )
        preservation["passes_rmse_preservation_90"] = preservation["rmse_preservation_vs_empirical"] >= 0.90
        preservation["passes_dist_preservation_90"] = preservation["dist_kl_preservation_vs_empirical"] >= 0.90
        preservation["stage2_success"] = (
            preservation["variant"].isin(["qcbm_prior", "finite_shot_qcbm_prior"])
            & preservation["beats_shuffled_rmse"]
            & preservation["beats_shuffled_dist_kl"]
            & (
                preservation["passes_rmse_preservation_90"]
                | preservation["passes_dist_preservation_90"]
            )
        )
    preservation.to_csv(out_dir / "preservation.csv", index=False)


def qprior_fit_report(empirical: PriorSpec, qcbm: PriorSpec, finite: PriorSpec, out_dir: Path):
    rows = []
    flat_emp = normalize_probs(empirical.probs.reshape(-1))
    for name, spec in [("qcbm_prior", qcbm), ("finite_shot_qcbm_prior", finite)]:
        flat = normalize_probs(spec.probs.reshape(-1))
        flat_metrics = compare_distributions(flat, flat_emp, np.arange(len(flat_emp), dtype=np.float64))
        row_metrics = [
            compare_distributions(spec.probs[i], empirical.probs[i], empirical.centers)
            for i in range(empirical.probs.shape[0])
        ]
        row = {
            "variant": name,
            **{f"flat_{k}": v for k, v in flat_metrics.items()},
        }
        for key in row_metrics[0]:
            row[f"mean_row_{key}"] = float(np.mean([m[key] for m in row_metrics]))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "qprior_fit.csv", index=False)


def run_stage2(args):
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.quick:
        seeds = [0]
        args.epochs = min(args.epochs, 2)

    cfg = ExperimentConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seeds=seeds,
        epochs=args.epochs,
        factors=args.factors,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        lambda_kl=args.lambda_kl,
        lambda_mmd=args.lambda_mmd,
        lambda_peak=args.lambda_peak,
        soft_tau=args.soft_tau,
        residual_bins=args.residual_bins,
        dot_bins=32,
        user_clusters=args.user_clusters,
        item_clusters=args.item_clusters,
        top_k=args.top_k,
        max_candidates=None,
        include_oracle=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, val, test, user_to_idx, item_to_idx, mu, empirical, qcbm, finite = prepare_data_and_priors(args, cfg)
    qprior_fit_report(empirical, qcbm, finite, args.out_dir)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate": args.candidate,
        "data_dir": str(args.data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_users": len(user_to_idx),
        "n_items": len(item_to_idx),
        "empirical_prior": str(args.empirical_prior),
        "qcbm_prior": str(args.qcbm_prior),
        "finite_shot_prior": str(args.finite_shot_prior),
        "prior_shape": list(empirical.probs.shape),
        "config": vars(cfg) | {"data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    metrics_rows = []
    history_rows = []
    subgroup_rows = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        initial = MatrixFactorization(len(user_to_idx), len(item_to_idx), cfg.factors, mu, seed)
        baseline = initial.copy()
        baseline, hist = train_model(baseline, train, val, cfg, prior=None, seed=seed)
        metrics_rows.append(
            collect_stage2_metrics("baseline", seed, baseline, train, val, test, None, empirical, cfg, len(item_to_idx))
        )
        for _, row in hist.iterrows():
            history_rows.append({"seed": seed, "variant": "baseline", **row.to_dict()})
        sg = subgroup_metrics(baseline, train, test)
        sg["seed"] = seed
        sg["variant"] = "baseline"
        subgroup_rows.extend(sg.to_dict("records"))

        variants = [
            ("empirical_prior", empirical),
            ("qcbm_prior", qcbm),
            ("finite_shot_qcbm_prior", finite),
            ("shuffled_prior", empirical.shuffled(np.random.default_rng(seed + 1001))),
            ("uniform_prior", empirical.uniform()),
            ("random_prior", empirical.random(np.random.default_rng(seed + 2002))),
        ]

        for variant, prior in variants:
            print(f"variant={variant}")
            model = initial.copy()
            model, hist = train_model(model, train, val, cfg, prior=prior, seed=seed)
            metrics_rows.append(
                collect_stage2_metrics(variant, seed, model, train, val, test, prior, empirical, cfg, len(item_to_idx))
            )
            for _, row in hist.iterrows():
                history_rows.append({"seed": seed, "variant": variant, **row.to_dict()})
            sg = subgroup_metrics(model, train, test)
            sg["seed"] = seed
            sg["variant"] = variant
            subgroup_rows.extend(sg.to_dict("records"))

        pd.DataFrame(metrics_rows).to_csv(args.out_dir / "metrics.csv", index=False)
        pd.DataFrame(history_rows).to_csv(args.out_dir / "history.csv", index=False)
        pd.DataFrame(subgroup_rows).to_csv(args.out_dir / "subgroups.csv", index=False)
        summarize_stage2(args.out_dir)

    return args.out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Re-test recommender with empirical vs trained QCBM priors.")
    parser.add_argument("--candidate", default="cond_user_itemtype")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument("--empirical_prior", type=Path, default=Path("results/qiml_prior_experiments/priors/cond_user_itemtype.npz"))
    parser.add_argument("--qcbm_prior", type=Path, default=Path("models/qprior_winning/cond_user_itemtype/qcbm_prior.npy"))
    parser.add_argument("--finite_shot_prior", type=Path, default=Path("models/qprior_winning/cond_user_itemtype/qcbm_prior_finite_shots.npy"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/qiml_qprior_retest/cond_user_itemtype"))
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--factors", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_mmd", type=float, default=0.05)
    parser.add_argument("--lambda_peak", type=float, default=0.0)
    parser.add_argument("--soft_tau", type=float, default=0.25)
    parser.add_argument("--residual_bins", type=int, default=16)
    parser.add_argument("--user_clusters", type=int, default=12)
    parser.add_argument("--item_clusters", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    out_dir = run_stage2(parse_args())
    print(f"\nWrote Q-prior recommender re-test outputs to {out_dir}")
    print("Key files: summary.csv, preservation.csv, qprior_fit.csv, metrics_with_gains.csv")


if __name__ == "__main__":
    main()
