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
    collect_all_metrics,
    compare_distributions,
    distribution_metrics,
    normalize_probs,
    subgroup_metrics,
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


def verify_qcbm_checkpoint_report(
    qcbm_report: Dict,
    learned_flat: np.ndarray,
    target_flat: np.ndarray,
) -> None:
    """Verify that the saved distribution is the reported best fit checkpoint."""

    if qcbm_report.get("checkpoint_policy") != "lowest post-update KL+L2 fitting loss":
        raise ValueError("QCBM report does not certify corrected post-update best checkpointing.")
    history = qcbm_report.get("history", [])
    if not history:
        raise ValueError("QCBM report has no fitting history to verify.")
    selected = [row for row in history if row.get("selected") is True]
    if len(selected) != 1:
        raise ValueError(f"QCBM report must mark one selected checkpoint, found {len(selected)}.")
    minimum = min(history, key=lambda row: float(row["loss"]))
    if int(minimum["epoch"]) != int(qcbm_report.get("best_epoch", -1)):
        raise ValueError("QCBM best_epoch is not the minimum post-update fitting-loss epoch.")
    if int(selected[0]["epoch"]) != int(minimum["epoch"]):
        raise ValueError("QCBM selected checkpoint does not match the minimum fitting-loss epoch.")
    if not np.isclose(
        float(qcbm_report.get("best_loss", np.nan)),
        float(minimum["loss"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("QCBM best_loss does not match the fitting history minimum.")

    learned_flat = normalize_probs(learned_flat)
    target_flat = normalize_probs(target_flat)
    eps = 1e-10
    fitted_loss = float(
        np.sum(learned_flat * np.log((learned_flat + eps) / (target_flat + eps)))
        + np.sum((learned_flat - target_flat) ** 2)
    )
    if not np.isclose(fitted_loss, float(minimum["loss"]), rtol=0.0, atol=1e-10):
        raise ValueError("Saved QCBM prior is not the reported minimum-loss checkpoint.")


def prepare_data_and_priors(args, cfg: ExperimentConfig):
    # Importing here avoids a module cycle while guaranteeing that final
    # confirmation rebuilds the exact search-stage grouping implementation.
    from svd_prior_search import (
        build_user_item_time_prior,
        build_user_itemtype_prior,
        compute_item_clusters,
        compute_user_clusters,
        load_context,
    )

    ctx = load_context(cfg)
    user_map = compute_user_clusters(ctx, args.user_clusters)
    item_map = compute_item_clusters(ctx, args.item_clusters)
    if args.candidate == "cond_user_itemtype":
        base = build_user_itemtype_prior(
            ctx,
            user_map,
            item_map,
            args.user_clusters,
            args.item_clusters,
            args.candidate,
        )
    elif args.candidate == "cond_user_item_time":
        if not args.time_scheme:
            raise ValueError("cond_user_item_time requires time_scheme in the selection manifest.")
        base = build_user_item_time_prior(
            ctx,
            user_map,
            item_map,
            args.user_clusters,
            args.item_clusters,
            args.time_scheme,
            args.candidate,
        )
    else:
        raise ValueError(f"Unsupported corrected static winner: {args.candidate!r}")

    saved = load_empirical_prior(args.empirical_prior)
    if tuple(saved["probs"].shape) != tuple(base.probs.shape):
        raise ValueError(
            f"Saved empirical prior shape {saved['probs'].shape} does not match rebuilt "
            f"{args.candidate} shape {base.probs.shape}."
        )
    if not np.allclose(saved["centers"], base.centers, rtol=0.0, atol=1e-12):
        raise ValueError("Saved empirical-prior centers do not match the corrected protocol split.")
    if not np.allclose(saved["probs"], base.probs, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Saved empirical prior was not built from this corrected protocol/configuration."
        )

    qcbm_target_path = args.qcbm_prior.parent / "target_prior.npy"
    qcbm_report_path = args.qcbm_prior.parent / "qcbm_report.json"
    if not qcbm_target_path.exists() or not qcbm_report_path.exists():
        raise ValueError("QCBM directory must contain target_prior.npy and qcbm_report.json.")
    expected_flat_target = normalize_probs(np.asarray(saved["probs"]).reshape(-1))
    if not np.allclose(np.load(qcbm_target_path), expected_flat_target, rtol=0.0, atol=1e-12):
        raise ValueError("QCBM was not trained against the selected empirical prior.")
    qcbm_report = json.loads(qcbm_report_path.read_text())
    learned_flat = np.load(args.qcbm_prior).astype(np.float64)
    verify_qcbm_checkpoint_report(qcbm_report, learned_flat, expected_flat_target)

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

    return (
        ctx.train,
        ctx.checkpoint_val,
        ctx.selection_val,
        ctx.confirmation,
        ctx.user_to_idx,
        ctx.item_to_idx,
        ctx.mu,
        empirical,
        qcbm,
        finite,
    )


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


def positive_gain_preservation(variant_gain: float, empirical_gain: float) -> float:
    """Return preserved benefit only when the empirical reference has a benefit.

    A ratio of two negative gains is positive but does not represent preserved
    improvement.  Treat that case as undefined so it cannot satisfy a
    preservation gate.
    """

    if not np.isfinite(empirical_gain) or empirical_gain <= 0.0:
        return np.nan
    return float(variant_gain) / float(empirical_gain)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 50_000):
    """Deterministic percentile interval for a paired mean difference."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_seed_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare frozen variants using paired initialization-seed differences.

    Positive differences mean the target variant is better.  These intervals
    measure sensitivity to the ten training/null seeds; they are not a
    population-level uncertainty estimate for a new ratings data set.
    """

    targets_and_comparators = {
        "empirical_prior": ["baseline", "shuffled_prior", "uniform_prior", "random_prior"],
        "qcbm_prior": [
            "baseline",
            "empirical_prior",
            "shuffled_prior",
            "uniform_prior",
            "random_prior",
        ],
        "finite_shot_qcbm_prior": [
            "baseline",
            "empirical_prior",
            "shuffled_prior",
            "uniform_prior",
            "random_prior",
        ],
    }
    rmse = metrics.pivot(index="seed", columns="variant", values="test_rmse")
    dist_kl = metrics.pivot(index="seed", columns="variant", values="empirical_dist_kl")
    rows = []
    comparison_seed = 20_260_915
    for target, comparators in targets_and_comparators.items():
        for comparator in comparators:
            if target not in rmse or comparator not in rmse:
                continue
            paired = pd.concat(
                [
                    rmse[comparator].rename("comparator_rmse"),
                    rmse[target].rename("target_rmse"),
                    dist_kl[comparator].rename("comparator_dist_kl"),
                    dist_kl[target].rename("target_dist_kl"),
                ],
                axis=1,
            ).dropna()
            rmse_improvement = (
                paired["comparator_rmse"] - paired["target_rmse"]
            ).to_numpy()
            dist_improvement = (
                paired["comparator_dist_kl"] - paired["target_dist_kl"]
            ).to_numpy()
            rmse_low, rmse_high = bootstrap_mean_ci(rmse_improvement, comparison_seed)
            dist_low, dist_high = bootstrap_mean_ci(dist_improvement, comparison_seed + 1)
            comparison_seed += 2
            rows.append(
                {
                    "target_variant": target,
                    "comparator_variant": comparator,
                    "paired_seeds": len(paired),
                    "mean_rmse_improvement": float(rmse_improvement.mean()),
                    "rmse_bootstrap_95_low": rmse_low,
                    "rmse_bootstrap_95_high": rmse_high,
                    "rmse_winning_seeds": int(np.sum(rmse_improvement > 0.0)),
                    "rmse_ci_excludes_zero": bool(rmse_low > 0.0),
                    "mean_dist_kl_improvement": float(dist_improvement.mean()),
                    "dist_kl_bootstrap_95_low": dist_low,
                    "dist_kl_bootstrap_95_high": dist_high,
                    "dist_kl_winning_seeds": int(np.sum(dist_improvement > 0.0)),
                    "dist_kl_ci_excludes_zero": bool(dist_low > 0.0),
                    "inference_scope": "paired training/null seeds only",
                }
            )
    return pd.DataFrame(rows)


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
    paired = paired_seed_comparisons(metrics)
    paired.to_csv(out_dir / "paired_seed_comparisons.csv", index=False)

    empirical = summary[summary["variant"] == "empirical_prior"].iloc[0]
    rows = []
    for variant in ["qcbm_prior", "finite_shot_qcbm_prior", "shuffled_prior", "uniform_prior", "random_prior"]:
        if variant not in set(summary["variant"]):
            continue
        row = summary[summary["variant"] == variant].iloc[0].to_dict()
        rmse_den = float(empirical["mean_rmse_gain"])
        dist_den = float(empirical["mean_dist_kl_gain"])
        row["empirical_has_positive_rmse_gain"] = rmse_den > 0.0
        row["empirical_has_positive_dist_kl_gain"] = dist_den > 0.0
        row["rmse_preservation_vs_empirical"] = positive_gain_preservation(
            float(row["mean_rmse_gain"]), rmse_den
        )
        row["dist_kl_preservation_vs_empirical"] = positive_gain_preservation(
            float(row["mean_dist_kl_gain"]), dist_den
        )
        row["beats_shuffled_rmse"] = np.nan
        row["beats_shuffled_dist_kl"] = np.nan
        rows.append(row)

    preservation = pd.DataFrame(rows)
    null_variants = ["shuffled_prior", "uniform_prior", "random_prior"]
    if not preservation.empty and all(v in set(summary["variant"]) for v in null_variants):
        for null_variant in null_variants:
            null = summary[summary["variant"] == null_variant].iloc[0]
            short = null_variant.removesuffix("_prior")
            preservation[f"beats_{short}_rmse"] = preservation["mean_rmse_gain"] > float(
                null["mean_rmse_gain"]
            )
            preservation[f"beats_{short}_dist_kl"] = preservation["mean_empirical_dist_kl"] < float(
                null["mean_empirical_dist_kl"]
            )
        preservation["beats_all_nulls_rmse"] = preservation[
            [f"beats_{v.removesuffix('_prior')}_rmse" for v in null_variants]
        ].all(axis=1)
        preservation["beats_all_nulls_dist_kl"] = preservation[
            [f"beats_{v.removesuffix('_prior')}_dist_kl" for v in null_variants]
        ].all(axis=1)
        preservation["passes_rmse_preservation_90"] = preservation["rmse_preservation_vs_empirical"] >= 0.90
        preservation["passes_dist_preservation_90"] = preservation["dist_kl_preservation_vs_empirical"] >= 0.90
        preservation["stage2_success"] = (
            preservation["variant"].isin(["qcbm_prior", "finite_shot_qcbm_prior"])
            & preservation["beats_all_nulls_rmse"]
            & preservation["beats_all_nulls_dist_kl"]
            & (
                preservation["passes_rmse_preservation_90"]
                | preservation["passes_dist_preservation_90"]
            )
        )
        paired_lookup = paired.set_index(["target_variant", "comparator_variant"])
        for endpoint in ("rmse", "dist_kl"):
            column = f"paired_seed_beats_all_nulls_{endpoint}_95"
            preservation[column] = preservation["variant"].map(
                lambda variant: all(
                    bool(paired_lookup.loc[(variant, null), f"{endpoint}_ci_excludes_zero"])
                    for null in null_variants
                    if (variant, null) in paired_lookup.index
                )
                and all((variant, null) in paired_lookup.index for null in null_variants)
            )
        preservation["seed_robust_stage2_success"] = (
            preservation["stage2_success"]
            & preservation["paired_seed_beats_all_nulls_rmse_95"]
            & preservation["paired_seed_beats_all_nulls_dist_kl_95"]
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


def apply_selection_manifest(args):
    """Load the configuration frozen before confirmation metrics were read."""

    if args.selection_manifest is None:
        raise ValueError(
            "Corrected Stage 2 requires --selection_manifest from validation-only Stage C."
        )
    selection = json.loads(args.selection_manifest.read_text())
    if selection.get("confirmation_metrics_accessed") is not False:
        raise ValueError("Selection manifest does not certify an untouched confirmation outcome.")
    if selection.get("qcbm_eligible") is not True:
        raise ValueError(
            "Validation-selected empirical prior did not beat every null control; "
            "the protocol forbids QCBM training/confirmation."
        )
    for name in (
        "candidate",
        "epochs",
        "factors",
        "lr",
        "l2",
        "lambda_kl",
        "lambda_mmd",
        "soft_tau",
        "user_clusters",
        "item_clusters",
    ):
        setattr(args, name, selection[name])
    args.time_scheme = selection.get("time_scheme")
    args.empirical_prior = Path(selection["empirical_prior"])
    return selection


def history_selected_epoch(history: pd.DataFrame) -> int:
    values = history.loc[history["selected"], "epoch"]
    if len(values) != 1:
        raise RuntimeError(f"Expected one validation-selected epoch, found {len(values)}.")
    return int(values.iloc[0])


def run_stage2(args):
    selection = apply_selection_manifest(args)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

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

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite completed/partial confirmation output at {args.out_dir}. "
            "Use a new output directory; confirmation is a one-shot protocol step."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (
        train,
        checkpoint_val,
        selection_val,
        test,
        user_to_idx,
        item_to_idx,
        mu,
        empirical,
        qcbm,
        finite,
    ) = prepare_data_and_priors(args, cfg)
    from svd_prior_search import split_fingerprint

    actual_fingerprints = {
        "train": split_fingerprint(train),
        "checkpoint_val": split_fingerprint(checkpoint_val),
        "selection_val": split_fingerprint(selection_val),
        "confirmation": split_fingerprint(test),
    }
    if actual_fingerprints != selection.get("split_fingerprints"):
        raise ValueError("Current data partitions do not match the frozen selection manifest.")
    qprior_fit_report(empirical, qcbm, finite, args.out_dir)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_version": "static_v1_corrected_nested_validation",
        "confirmation_policy": "configuration frozen before one-shot confirmation evaluation",
        "selection_manifest": str(args.selection_manifest),
        "selection": selection,
        "verified_split_fingerprints": actual_fingerprints,
        "candidate": args.candidate,
        "data_dir": str(args.data_dir),
        "n_train": len(train),
        "n_checkpoint_val": len(checkpoint_val),
        "n_selection_val": len(selection_val),
        "n_confirmation": len(test),
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
        baseline, hist = train_model(
            baseline, train, checkpoint_val, cfg, prior=None, seed=seed
        )
        metric = collect_stage2_metrics(
            "baseline", seed, baseline, train, checkpoint_val, test, None, empirical, cfg, len(item_to_idx)
        )
        metric["selected_epoch"] = history_selected_epoch(hist)
        metrics_rows.append(metric)
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
            model, hist = train_model(
                model, train, checkpoint_val, cfg, prior=prior, seed=seed
            )
            metric = collect_stage2_metrics(
                variant,
                seed,
                model,
                train,
                checkpoint_val,
                test,
                prior,
                empirical,
                cfg,
                len(item_to_idx),
            )
            metric["selected_epoch"] = history_selected_epoch(hist)
            metrics_rows.append(metric)
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
    parser.add_argument(
        "--selection_manifest",
        type=Path,
        required=True,
        help="Validation-only configuration freeze emitted by svd_prior_search.py Stage C.",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument(
        "--empirical_prior",
        type=Path,
        default=Path(
            "results/static_v1_corrected/svd_prior_search_common_reference/"
            "best_config/selected/priors/cond_user_itemtype.npz"
        ),
    )
    parser.add_argument(
        "--qcbm_prior",
        type=Path,
        default=Path(
            "models/qprior_winning/static_v1_corrected/cond_user_itemtype/qcbm_prior.npy"
        ),
    )
    parser.add_argument(
        "--finite_shot_prior",
        type=Path,
        default=Path(
            "models/qprior_winning/static_v1_corrected/cond_user_itemtype/"
            "qcbm_prior_finite_shots.npy"
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("results/static_v1_corrected/qiml_qprior_confirmation"),
    )
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
    return parser.parse_args()


def main():
    out_dir = run_stage2(parse_args())
    print(f"\nWrote Q-prior recommender re-test outputs to {out_dir}")
    print(
        "Key files: summary.csv, preservation.csv, paired_seed_comparisons.csv, "
        "qprior_fit.csv, metrics_with_gains.csv"
    )


if __name__ == "__main__":
    main()
