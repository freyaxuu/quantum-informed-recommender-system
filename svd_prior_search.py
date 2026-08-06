#!/usr/bin/env python3
"""Stage 1.5: standard SVD/MF tabulation plus a staged hyperparameter and
grouping/method search for the static prior-loss recommender.

This script does not duplicate model or metric logic.  It imports the MF
model, training loop, prior machinery, and metric functions directly from
`qiml_prior_experiments.py` and only adds:

  1. A grid tabulation of the plain (no-prior) MF/SVD model, to establish
     what "standard SVD" actually achieves on this split before any prior
     loss is added.
  2. A new time-of-day / weekday-weekend grouping feature, which does not
     exist anywhere else in the repository (timestamps are otherwise only
     used for chronological ordering).
  3. A staged (cheap -> moderate -> full-cost) search over user-cluster
     count, item-cluster count, the new time axis, and prior-loss weights,
     using the same paired-seed / shuffled-control screening approach and
     "meaningful_candidate" bar as `qiml_prior_experiments.summarize_results`.

Usage:

    python3 svd_prior_search.py --quick
    python3 svd_prior_search.py --stage baseline
    python3 svd_prior_search.py --stage a
    python3 svd_prior_search.py --stage b
    python3 svd_prior_search.py --stage c
    python3 svd_prior_search.py --stage all
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qiml_prior_experiments import (
    ExperimentConfig,
    MatrixFactorization,
    PriorSpec,
    add_indices,
    add_stat_columns,
    attach_group_column,
    build_item_genres,
    build_item_type_features,
    build_user_residual_features,
    collect_all_metrics,
    fit_stat_biases,
    grouped_residual_probs,
    histogram,
    load_movielens,
    prior_summary_row,
    ranking_metrics,
    rating_metrics,
    robust_quantile_bins,
    save_prior_spec,
    simple_kmeans,
    summarize_results,
    temporal_split,
    train_model,
)

TIME_SCHEME_BUCKETS = {"tod3": 3, "weekday_weekend": 2}


# --------------------------------------------------------------------------
# Time-bucket feature (new)
# --------------------------------------------------------------------------


def build_time_bucket(df: pd.DataFrame, scheme: str) -> pd.Series:
    """Deterministic time bucket from raw MovieLens Unix timestamps.

    MovieLens timestamps carry no recorded timezone, so "tod3" buckets
    epoch-UTC hour rather than each user's true local time of day -- an
    honest caveat, not a correctness claim.  The bucket is a pure function
    of `timestamp`, so it can be computed independently on train/val/test
    with no target leakage.
    """
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    if scheme == "tod3":
        hour = (ts // 3600) % 24
        bucket = np.where((hour >= 5) & (hour <= 11), 0, np.where((hour >= 12) & (hour <= 17), 1, 2))
    elif scheme == "weekday_weekend":
        weekday = np.array([datetime.fromtimestamp(int(t), tz=timezone.utc).weekday() for t in ts])
        bucket = np.where(weekday >= 5, 1, 0)
    else:
        raise ValueError(f"Unknown time scheme: {scheme}")
    return pd.Series(bucket, index=df.index, dtype=int)


def attach_group_column3(
    splits,
    name: str,
    group_map: Dict[Tuple[int, int, int], int],
    key_cols: Tuple[str, str, str],
) -> None:
    """Three-key generalization of qiml_prior_experiments.attach_group_column."""
    col = f"{name}_group_idx"
    for df in splits:
        df[col] = [
            group_map.get((int(a), int(b), int(c)), 0)
            for a, b, c in zip(
                df[key_cols[0]].to_numpy(),
                df[key_cols[1]].to_numpy(),
                df[key_cols[2]].to_numpy(),
            )
        ]


# --------------------------------------------------------------------------
# Shared data/context
# --------------------------------------------------------------------------


@dataclass
class SearchContext:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    user_to_idx: Dict[int, int]
    item_to_idx: Dict[int, int]
    item_genres: Dict[int, List[int]]
    item_genre_matrix: np.ndarray
    n_genres: int
    mu: float
    residual_centers: np.ndarray
    residual_edges: np.ndarray
    global_resid: np.ndarray


def load_context(cfg: ExperimentConfig) -> SearchContext:
    ratings, items = load_movielens(cfg.data_dir)
    train, val, test = temporal_split(ratings)
    train, val, test, user_to_idx, item_to_idx = add_indices(train, val, test)
    mu = float(train["rating"].mean())
    user_bias, item_bias = fit_stat_biases(train, mu)
    train = add_stat_columns(train, user_bias, item_bias, mu)
    val = add_stat_columns(val, user_bias, item_bias, mu)
    test = add_stat_columns(test, user_bias, item_bias, mu)
    genre_cols, item_genres, item_genre_matrix = build_item_genres(items, item_to_idx)
    residual_edges, residual_centers = robust_quantile_bins(
        train["residual_stat"].to_numpy(), cfg.residual_bins
    )
    global_resid = histogram(train["residual_stat"].to_numpy(), residual_centers, residual_edges)
    return SearchContext(
        train=train,
        val=val,
        test=test,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        item_genres=item_genres,
        item_genre_matrix=item_genre_matrix,
        n_genres=len(genre_cols),
        mu=mu,
        residual_centers=residual_centers,
        residual_edges=residual_edges,
        global_resid=global_resid,
    )


def compute_user_clusters(ctx: SearchContext, k: int, seed: int = 42) -> Dict[int, int]:
    users, features, _ = build_user_residual_features(ctx.train, ctx.item_genres, ctx.n_genres)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    x = (features - features.mean(axis=0)) / std
    labels = simple_kmeans(x, k, seed=seed)
    return {int(u): int(label) for u, label in zip(users, labels)}


def compute_item_clusters(ctx: SearchContext, k: int, seed: int = 42) -> Dict[int, int]:
    item_x = build_item_type_features(ctx.train, len(ctx.item_to_idx), ctx.item_genre_matrix)
    labels = simple_kmeans(item_x, k, seed=seed)
    idx_to_item = {idx: item_id for item_id, idx in ctx.item_to_idx.items()}
    return {int(idx_to_item[idx]): int(label) for idx, label in enumerate(labels)}


# --------------------------------------------------------------------------
# Prior-spec builders (search candidates)
# --------------------------------------------------------------------------


def build_user_itemtype_prior(
    ctx: SearchContext, user_map: Dict[int, int], item_map: Dict[int, int], k_user: int, k_item: int, name: str
) -> PriorSpec:
    for df in (ctx.train, ctx.val, ctx.test):
        df["user_cluster"] = df["user_id"].map(user_map).fillna(0).astype(int)
        df["item_cluster"] = df["item_id"].map(item_map).fillna(0).astype(int)
    groups = [(u, i) for u in range(k_user) for i in range(k_item)]
    group_map = {g: idx for idx, g in enumerate(groups)}
    attach_group_column((ctx.train, ctx.val, ctx.test), name, group_map, ("user_cluster", "item_cluster"))
    probs = grouped_residual_probs(
        ctx.train, f"{name}_group_idx", len(groups), ctx.residual_centers, ctx.residual_edges, fallback=ctx.global_resid
    )
    return PriorSpec(
        name,
        "residual",
        ctx.residual_centers,
        probs,
        "sample",
        ["user_cluster", "item_cluster"],
        f"P(E | user_cluster k={k_user}, item_cluster k={k_item})",
        group_col=f"{name}_group_idx",
    )


def build_user_time_prior(
    ctx: SearchContext, user_map: Dict[int, int], k_user: int, time_scheme: str, name: str
) -> PriorSpec:
    for df in (ctx.train, ctx.val, ctx.test):
        df["user_cluster"] = df["user_id"].map(user_map).fillna(0).astype(int)
        df["time_bucket"] = build_time_bucket(df, time_scheme)
    n_time = TIME_SCHEME_BUCKETS[time_scheme]
    groups = [(u, t) for u in range(k_user) for t in range(n_time)]
    group_map = {g: idx for idx, g in enumerate(groups)}
    attach_group_column((ctx.train, ctx.val, ctx.test), name, group_map, ("user_cluster", "time_bucket"))
    probs = grouped_residual_probs(
        ctx.train, f"{name}_group_idx", len(groups), ctx.residual_centers, ctx.residual_edges, fallback=ctx.global_resid
    )
    return PriorSpec(
        name,
        "residual",
        ctx.residual_centers,
        probs,
        "sample",
        ["user_cluster", "time_bucket"],
        f"P(E | user_cluster k={k_user}, time={time_scheme})",
        group_col=f"{name}_group_idx",
    )


def build_item_time_prior(
    ctx: SearchContext, item_map: Dict[int, int], k_item: int, time_scheme: str, name: str
) -> PriorSpec:
    for df in (ctx.train, ctx.val, ctx.test):
        df["item_cluster"] = df["item_id"].map(item_map).fillna(0).astype(int)
        df["time_bucket"] = build_time_bucket(df, time_scheme)
    n_time = TIME_SCHEME_BUCKETS[time_scheme]
    groups = [(i, t) for i in range(k_item) for t in range(n_time)]
    group_map = {g: idx for idx, g in enumerate(groups)}
    attach_group_column((ctx.train, ctx.val, ctx.test), name, group_map, ("item_cluster", "time_bucket"))
    probs = grouped_residual_probs(
        ctx.train, f"{name}_group_idx", len(groups), ctx.residual_centers, ctx.residual_edges, fallback=ctx.global_resid
    )
    return PriorSpec(
        name,
        "residual",
        ctx.residual_centers,
        probs,
        "sample",
        ["item_cluster", "time_bucket"],
        f"P(E | item_cluster k={k_item}, time={time_scheme})",
        group_col=f"{name}_group_idx",
    )


def build_user_item_time_prior(
    ctx: SearchContext,
    user_map: Dict[int, int],
    item_map: Dict[int, int],
    k_user: int,
    k_item: int,
    time_scheme: str,
    name: str,
) -> PriorSpec:
    for df in (ctx.train, ctx.val, ctx.test):
        df["user_cluster"] = df["user_id"].map(user_map).fillna(0).astype(int)
        df["item_cluster"] = df["item_id"].map(item_map).fillna(0).astype(int)
        df["time_bucket"] = build_time_bucket(df, time_scheme)
    n_time = TIME_SCHEME_BUCKETS[time_scheme]
    groups = [(u, i, t) for u in range(k_user) for i in range(k_item) for t in range(n_time)]
    group_map = {g: idx for idx, g in enumerate(groups)}
    attach_group_column3(
        (ctx.train, ctx.val, ctx.test), name, group_map, ("user_cluster", "item_cluster", "time_bucket")
    )
    probs = grouped_residual_probs(
        ctx.train, f"{name}_group_idx", len(groups), ctx.residual_centers, ctx.residual_edges, fallback=ctx.global_resid
    )
    return PriorSpec(
        name,
        "residual",
        ctx.residual_centers,
        probs,
        "sample",
        ["user_cluster", "item_cluster", "time_bucket"],
        f"P(E | user_cluster k={k_user}, item_cluster k={k_item}, time={time_scheme})",
        group_col=f"{name}_group_idx",
    )


# --------------------------------------------------------------------------
# Candidate evaluation
# --------------------------------------------------------------------------


def train_baseline_pool(
    ctx: SearchContext, cfg: ExperimentConfig, seeds: Sequence[int]
) -> Dict[int, Tuple[MatrixFactorization, Dict]]:
    """Train the no-prior baseline once per seed for a fixed (factors, lr, l2,
    batch_size, epochs).  Baseline training never touches the prior, so within
    a stage where only the prior/grouping or only lambda_kl/lambda_mmd/soft_tau
    changes, this is safe to reuse instead of retraining per candidate."""
    n_users = len(ctx.user_to_idx)
    n_items = len(ctx.item_to_idx)
    pool: Dict[int, Tuple[MatrixFactorization, Dict]] = {}
    for seed in seeds:
        initial = MatrixFactorization(n_users, n_items, cfg.factors, ctx.mu, seed)
        baseline = initial.copy()
        baseline, _ = train_model(baseline, ctx.train, ctx.val, cfg, prior=None, seed=seed)
        base_row = collect_all_metrics("baseline", "none", seed, baseline, ctx.train, ctx.val, ctx.test, None, cfg, n_items)
        pool[seed] = (initial, base_row)
    return pool


def evaluate_candidate(
    ctx: SearchContext,
    cfg: ExperimentConfig,
    prior: PriorSpec,
    seeds: Sequence[int],
    variants: Sequence[str] = ("real_prior", "shuffled_prior"),
    baseline_pool: Optional[Dict[int, Tuple[MatrixFactorization, Dict]]] = None,
) -> Dict[str, float]:
    """Train (or reuse a cached) baseline plus the requested prior variants
    across seeds; return mean RMSE gain / distribution KL for each requested
    variant, paired against the per-seed baseline (mirrors run_campaign's
    baseline-then-variants loop in qiml_prior_experiments.py, but scoped to
    one candidate). `baseline_pool` must come from `train_baseline_pool`
    called with a cfg sharing the same factors/lr/l2/batch_size/epochs."""
    n_users = len(ctx.user_to_idx)
    n_items = len(ctx.item_to_idx)
    rows: List[Dict] = []
    for seed in seeds:
        if baseline_pool is not None and seed in baseline_pool:
            initial, base_row = baseline_pool[seed]
        else:
            initial = MatrixFactorization(n_users, n_items, cfg.factors, ctx.mu, seed)
            baseline = initial.copy()
            baseline, _ = train_model(baseline, ctx.train, ctx.val, cfg, prior=None, seed=seed)
            base_row = collect_all_metrics("baseline", "none", seed, baseline, ctx.train, ctx.val, ctx.test, None, cfg, n_items)
        rows.append(base_row)

        variant_specs = []
        if "real_prior" in variants:
            variant_specs.append(("real_prior", prior))
        if "shuffled_prior" in variants:
            variant_specs.append(("shuffled_prior", prior.shuffled(np.random.default_rng(seed + 1001))))
        if "uniform_prior" in variants:
            variant_specs.append(("uniform_prior", prior.uniform()))
        if "random_prior" in variants:
            variant_specs.append(("random_prior", prior.random(np.random.default_rng(seed + 2002))))

        for variant_name, spec in variant_specs:
            model = initial.copy()
            model, _ = train_model(model, ctx.train, ctx.val, cfg, prior=spec, seed=seed)
            rows.append(
                collect_all_metrics(
                    variant_name, prior.name, seed, model, ctx.train, ctx.val, ctx.test, spec, cfg, n_items
                )
            )

    df = pd.DataFrame(rows)
    base = df[df["variant"] == "baseline"][["seed", "test_rmse"]].rename(columns={"test_rmse": "baseline_test_rmse"})
    result: Dict[str, float] = {}
    for variant_name in variants:
        vdf = df[df["variant"] == variant_name].merge(base, on="seed", how="left")
        if vdf.empty:
            continue
        result[f"{variant_name}_rmse_gain"] = float((vdf["baseline_test_rmse"] - vdf["test_rmse"]).mean())
        result[f"{variant_name}_dist_kl"] = float(vdf["dist_kl"].mean())
    return result


def log_row(
    stage: str,
    config_id: str,
    prior: PriorSpec,
    cfg: ExperimentConfig,
    seeds: Sequence[int],
    eval_result: Dict[str, float],
    extra: Optional[Dict] = None,
) -> Dict:
    """Build one search_log.csv row.  Reuses the exact meaningful_candidate
    formula from qiml_prior_experiments.summarize_results (beats shuffled
    on both RMSE gain and dist KL, real gain >= -0.002)."""
    row = {
        "stage": stage,
        "config_id": config_id,
        "candidate": prior.name,
        "description": prior.description,
        "factors": cfg.factors,
        "lr": cfg.lr,
        "l2": cfg.l2,
        "lambda_kl": cfg.lambda_kl,
        "lambda_mmd": cfg.lambda_mmd,
        "soft_tau": cfg.soft_tau,
        "seeds": len(seeds),
    }
    if extra:
        row.update(extra)

    real_gain = eval_result.get("real_prior_rmse_gain")
    real_kl = eval_result.get("real_prior_dist_kl")
    shuf_gain = eval_result.get("shuffled_prior_rmse_gain")
    shuf_kl = eval_result.get("shuffled_prior_dist_kl")
    row["mean_rmse_gain"] = real_gain
    row["mean_dist_kl"] = real_kl
    row["shuffled_rmse_gain"] = shuf_gain
    row["shuffled_dist_kl"] = shuf_kl

    if real_gain is not None and shuf_gain is not None:
        beats_rmse = real_gain > (shuf_gain if shuf_gain is not None else -np.inf)
        beats_dist = real_kl < (shuf_kl if shuf_kl is not None else np.inf)
        row["beats_shuffled_rmse"] = bool(beats_rmse)
        row["beats_shuffled_dist"] = bool(beats_dist)
        row["meaningful_candidate"] = bool(beats_rmse and beats_dist and real_gain >= -0.002)
    else:
        row["beats_shuffled_rmse"] = None
        row["beats_shuffled_dist"] = None
        row["meaningful_candidate"] = None
    return row


def pick_best(rows: List[Dict]) -> Dict:
    """Prefer meaningful candidates; then higher RMSE gain; then lower dist KL."""
    df = pd.DataFrame(rows)
    df["meaningful_candidate"] = df["meaningful_candidate"].fillna(False)
    df = df.sort_values(
        ["meaningful_candidate", "mean_rmse_gain", "mean_dist_kl"], ascending=[False, False, True]
    )
    return df.iloc[0].to_dict()


def append_log(out_dir: Path, rows: List[Dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "search_log.csv"
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Stage 0: standard SVD tabulation
# --------------------------------------------------------------------------


def run_stage_baseline(
    ctx: SearchContext,
    base_cfg: ExperimentConfig,
    factor_grid: Sequence[int],
    lr_grid: Sequence[float],
    l2_grid: Sequence[float],
    seeds: Sequence[int],
    out_dir: Path,
) -> pd.DataFrame:
    n_users = len(ctx.user_to_idx)
    n_items = len(ctx.item_to_idx)
    metrics_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for factors, lr, l2 in itertools.product(factor_grid, lr_grid, l2_grid):
        cfg = replace(base_cfg, factors=factors, lr=lr, l2=l2)
        per_seed_rows: List[Dict] = []
        for seed in seeds:
            model = MatrixFactorization(n_users, n_items, factors, ctx.mu, seed)
            model, _ = train_model(model, ctx.train, ctx.val, cfg, prior=None, seed=seed)
            row = {"factors": factors, "lr": lr, "l2": l2, "seed": seed}
            row.update({f"val_{k}": v for k, v in rating_metrics(model, ctx.val).items()})
            row.update({f"test_{k}": v for k, v in rating_metrics(model, ctx.test).items()})
            row.update(ranking_metrics(model, ctx.train, ctx.test, n_items, cfg.top_k))
            metrics_rows.append(row)
            per_seed_rows.append(row)
        seed_df = pd.DataFrame(per_seed_rows)
        summary_rows.append(
            {
                "factors": factors,
                "lr": lr,
                "l2": l2,
                "seeds": len(seeds),
                "mean_val_rmse": seed_df["val_rmse"].mean(),
                "std_val_rmse": seed_df["val_rmse"].std(),
                "mean_test_rmse": seed_df["test_rmse"].mean(),
                "std_test_rmse": seed_df["test_rmse"].std(),
                "mean_test_mae": seed_df["test_mae"].mean(),
                "mean_recall_at_k": seed_df["recall_at_k"].mean(),
                "mean_ndcg_at_k": seed_df["ndcg_at_k"].mean(),
            }
        )
        print(
            f"[baseline] factors={factors} lr={lr} l2={l2} "
            f"mean_test_rmse={seed_df['test_rmse'].mean():.6f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("mean_test_rmse").reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


# --------------------------------------------------------------------------
# Stage A: cheap axis screen
# --------------------------------------------------------------------------


def run_stage_a(
    ctx: SearchContext,
    base_cfg: ExperimentConfig,
    seeds: Sequence[int],
    out_dir: Path,
    user_cluster_grid: Sequence[int],
    item_cluster_grid: Sequence[int],
    time_schemes: Sequence[str],
    default_user_clusters: int,
    default_item_clusters: int,
) -> Dict:
    log_rows: List[Dict] = []
    baseline_pool = train_baseline_pool(ctx, base_cfg, seeds)

    # 1) user_clusters sweep, item_clusters fixed at default.
    user_rows = []
    for k_user in user_cluster_grid:
        user_map = compute_user_clusters(ctx, k_user)
        item_map = compute_item_clusters(ctx, default_item_clusters)
        prior = build_user_itemtype_prior(ctx, user_map, item_map, k_user, default_item_clusters, "cond_user_itemtype")
        result = evaluate_candidate(ctx, base_cfg, prior, seeds, baseline_pool=baseline_pool)
        row = log_row(
            "A_user_clusters",
            f"user_clusters={k_user}",
            prior,
            base_cfg,
            seeds,
            result,
            {"user_clusters": k_user, "item_clusters": default_item_clusters, "time_scheme": None},
        )
        user_rows.append(row)
        log_rows.append(row)
        print(f"[A user_clusters={k_user}] gain={row['mean_rmse_gain']:.6f} shuf={row['shuffled_rmse_gain']:.6f}")
    best_user = pick_best(user_rows)
    best_user_clusters = int(best_user["user_clusters"])

    # 2) item_clusters sweep, user_clusters fixed at the Part-1 winner.
    item_rows = []
    user_map_best = compute_user_clusters(ctx, best_user_clusters)
    for k_item in item_cluster_grid:
        item_map = compute_item_clusters(ctx, k_item)
        prior = build_user_itemtype_prior(
            ctx, user_map_best, item_map, best_user_clusters, k_item, "cond_user_itemtype"
        )
        result = evaluate_candidate(ctx, base_cfg, prior, seeds, baseline_pool=baseline_pool)
        row = log_row(
            "A_item_clusters",
            f"item_clusters={k_item}",
            prior,
            base_cfg,
            seeds,
            result,
            {"user_clusters": best_user_clusters, "item_clusters": k_item, "time_scheme": None},
        )
        item_rows.append(row)
        log_rows.append(row)
        print(f"[A item_clusters={k_item}] gain={row['mean_rmse_gain']:.6f} shuf={row['shuffled_rmse_gain']:.6f}")
    best_item = pick_best(item_rows)
    best_item_clusters = int(best_item["item_clusters"])

    # 3) time screen: cond_user_time and cond_item_time, each time scheme.
    item_map_best = compute_item_clusters(ctx, best_item_clusters)
    time_rows = []
    for time_scheme in time_schemes:
        prior_ut = build_user_time_prior(ctx, user_map_best, best_user_clusters, time_scheme, "cond_user_time")
        result_ut = evaluate_candidate(ctx, base_cfg, prior_ut, seeds, baseline_pool=baseline_pool)
        row_ut = log_row(
            "A_time",
            f"user_time={time_scheme}",
            prior_ut,
            base_cfg,
            seeds,
            result_ut,
            {"user_clusters": best_user_clusters, "item_clusters": None, "time_scheme": time_scheme, "time_axis": "user"},
        )
        time_rows.append(row_ut)
        log_rows.append(row_ut)
        print(f"[A user_time={time_scheme}] gain={row_ut['mean_rmse_gain']:.6f} shuf={row_ut['shuffled_rmse_gain']:.6f}")

        prior_it = build_item_time_prior(ctx, item_map_best, best_item_clusters, time_scheme, "cond_item_time")
        result_it = evaluate_candidate(ctx, base_cfg, prior_it, seeds, baseline_pool=baseline_pool)
        row_it = log_row(
            "A_time",
            f"item_time={time_scheme}",
            prior_it,
            base_cfg,
            seeds,
            result_it,
            {"user_clusters": None, "item_clusters": best_item_clusters, "time_scheme": time_scheme, "time_axis": "item"},
        )
        time_rows.append(row_it)
        log_rows.append(row_it)
        print(f"[A item_time={time_scheme}] gain={row_it['mean_rmse_gain']:.6f} shuf={row_it['shuffled_rmse_gain']:.6f}")

    promising = [r for r in time_rows if r.get("beats_shuffled_rmse") and r.get("beats_shuffled_dist")]
    time_promising = len(promising) > 0
    best_time_scheme = None
    if time_promising:
        best_time_row = pick_best(promising)
        best_time_scheme = best_time_row["time_scheme"]

    append_log(out_dir, log_rows)

    winners = {
        "user_clusters": best_user_clusters,
        "item_clusters": best_item_clusters,
        "time_promising": time_promising,
        "time_scheme": best_time_scheme,
    }
    (out_dir / "stage_a_winners.json").write_text(json.dumps(winners, indent=2))
    print(f"[A] winners: {winners}")
    return winners


# --------------------------------------------------------------------------
# Stage B: combine grouping winner(s) with prior-loss weight tuning
# --------------------------------------------------------------------------


def run_stage_b(
    ctx: SearchContext,
    base_cfg: ExperimentConfig,
    seeds: Sequence[int],
    out_dir: Path,
    winners: Dict,
    lambda_kl_grid: Sequence[float],
    lambda_mmd_grid: Sequence[float],
    soft_tau_grid: Sequence[float],
) -> List[Dict]:
    k_user = winners["user_clusters"]
    k_item = winners["item_clusters"]
    user_map = compute_user_clusters(ctx, k_user)
    item_map = compute_item_clusters(ctx, k_item)

    grouping_specs: List[Tuple[str, PriorSpec]] = [
        ("cond_user_itemtype", build_user_itemtype_prior(ctx, user_map, item_map, k_user, k_item, "cond_user_itemtype"))
    ]
    if winners.get("time_promising") and winners.get("time_scheme"):
        time_scheme = winners["time_scheme"]
        grouping_specs.append(
            (
                "cond_user_item_time",
                build_user_item_time_prior(ctx, user_map, item_map, k_user, k_item, time_scheme, "cond_user_item_time"),
            )
        )

    # lambda_kl/lambda_mmd/soft_tau never affect baseline (prior=None) training,
    # and factors/lr/l2/batch_size/epochs are fixed at base_cfg throughout Stage B,
    # so one baseline pool covers every grid point below.
    baseline_pool = train_baseline_pool(ctx, base_cfg, seeds)

    log_rows: List[Dict] = []
    for grouping_name, base_prior in grouping_specs:
        for lambda_kl, lambda_mmd, soft_tau in itertools.product(lambda_kl_grid, lambda_mmd_grid, soft_tau_grid):
            cfg = replace(base_cfg, lambda_kl=lambda_kl, lambda_mmd=lambda_mmd, soft_tau=soft_tau)
            prior = base_prior
            result = evaluate_candidate(ctx, cfg, prior, seeds, baseline_pool=baseline_pool)
            config_id = f"{grouping_name}_kl{lambda_kl}_mmd{lambda_mmd}_tau{soft_tau}"
            row = log_row(
                "B",
                config_id,
                prior,
                cfg,
                seeds,
                result,
                {"user_clusters": k_user, "item_clusters": k_item, "time_scheme": winners.get("time_scheme")},
            )
            log_rows.append(row)
            print(
                f"[B {grouping_name} kl={lambda_kl} mmd={lambda_mmd} tau={soft_tau}] "
                f"gain={row['mean_rmse_gain']:.6f} shuf={row['shuffled_rmse_gain']:.6f}"
            )

    append_log(out_dir, log_rows)
    return log_rows


# --------------------------------------------------------------------------
# Stage C: full-cost confirmation of the top Stage-B configs
# --------------------------------------------------------------------------


def run_stage_c(
    ctx: SearchContext,
    base_cfg: ExperimentConfig,
    stage_b_rows: List[Dict],
    winners: Dict,
    seeds: Sequence[int],
    out_root: Path,
    top_k: int = 3,
) -> List[Dict]:
    df = pd.DataFrame(stage_b_rows)
    df["meaningful_candidate"] = df["meaningful_candidate"].fillna(False)
    df = df.sort_values(
        ["meaningful_candidate", "mean_rmse_gain", "mean_dist_kl"], ascending=[False, False, True]
    ).reset_index(drop=True)
    top_rows = df.head(top_k).to_dict("records")

    k_user = winners["user_clusters"]
    k_item = winners["item_clusters"]
    user_map = compute_user_clusters(ctx, k_user)
    item_map = compute_item_clusters(ctx, k_item)

    n_users = len(ctx.user_to_idx)
    n_items = len(ctx.item_to_idx)
    confirmed_rows: List[Dict] = []

    for rank, row in enumerate(top_rows, start=1):
        candidate_name = row["candidate"]
        cfg = replace(base_cfg, lambda_kl=row["lambda_kl"], lambda_mmd=row["lambda_mmd"], soft_tau=row["soft_tau"])
        if candidate_name == "cond_user_item_time":
            prior = build_user_item_time_prior(
                ctx, user_map, item_map, k_user, k_item, winners["time_scheme"], "cond_user_item_time"
            )
        else:
            prior = build_user_itemtype_prior(ctx, user_map, item_map, k_user, k_item, "cond_user_itemtype")

        out_dir = out_root / f"rank{rank}_{candidate_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_rows: List[Dict] = []
        for seed in seeds:
            initial = MatrixFactorization(n_users, n_items, cfg.factors, ctx.mu, seed)
            baseline = initial.copy()
            baseline, _ = train_model(baseline, ctx.train, ctx.val, cfg, prior=None, seed=seed)
            metrics_rows.append(
                collect_all_metrics("baseline", "none", seed, baseline, ctx.train, ctx.val, ctx.test, None, cfg, n_items)
            )
            variants = [
                ("real_prior", prior),
                ("shuffled_prior", prior.shuffled(np.random.default_rng(seed + 1001))),
                ("uniform_prior", prior.uniform()),
                ("random_prior", prior.random(np.random.default_rng(seed + 2002))),
            ]
            for variant_name, spec in variants:
                model = initial.copy()
                model, _ = train_model(model, ctx.train, ctx.val, cfg, prior=spec, seed=seed)
                metrics_rows.append(
                    collect_all_metrics(
                        variant_name, prior.name, seed, model, ctx.train, ctx.val, ctx.test, spec, cfg, n_items
                    )
                )

        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame([prior_summary_row(prior)]).to_csv(out_dir / "prior_catalog.csv", index=False)
        save_prior_spec(prior, out_dir / "priors")
        summarize_results(out_dir)

        winner_path = out_dir / "winner_screen.csv"
        if winner_path.exists():
            wdf = pd.read_csv(winner_path)
            wdf.insert(0, "rank", rank)
            wdf.insert(1, "config_dir", str(out_dir))
            wdf["factors"] = cfg.factors
            wdf["lr"] = cfg.lr
            wdf["l2"] = cfg.l2
            wdf["lambda_kl"] = cfg.lambda_kl
            wdf["lambda_mmd"] = cfg.lambda_mmd
            wdf["soft_tau"] = cfg.soft_tau
            confirmed_rows.extend(wdf.to_dict("records"))
        print(f"[C] rank {rank} ({candidate_name}) confirmed -> {out_dir}")

    if confirmed_rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(confirmed_rows).to_csv(out_root / "top_candidates_winner_screen.csv", index=False)
    return confirmed_rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Standard SVD tabulation + grouping/time hyperparameter search.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/svd_prior_search"))
    parser.add_argument("--baseline_out_dir", type=Path, default=Path("results/svd_baseline_grid"))
    parser.add_argument("--stage", choices=["baseline", "a", "b", "c", "all"], default="all")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--residual_bins", type=int, default=16)
    parser.add_argument("--dot_bins", type=int, default=32)

    parser.add_argument("--baseline_seeds", default="0,1,2,3,4")
    parser.add_argument("--stage_a_seeds", default="0,1,2")
    parser.add_argument("--stage_b_seeds", default="0,1,2,3,4")
    parser.add_argument("--stage_c_seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--stage_c_top", type=int, default=3)

    parser.add_argument("--factor_grid", default="20,50,100")
    parser.add_argument("--lr_grid", default="0.001,0.003,0.01")
    parser.add_argument("--l2_grid", default="1e-6,1e-5,1e-4")

    parser.add_argument("--user_cluster_grid", default="4,8,12,16,24")
    parser.add_argument("--item_cluster_grid", default="4,8,16,24,32")
    parser.add_argument("--time_schemes", default="tod3,weekday_weekend")
    parser.add_argument("--default_user_clusters", type=int, default=12)
    parser.add_argument("--default_item_clusters", type=int, default=16)

    parser.add_argument("--lambda_kl_grid", default="0.003,0.01,0.03")
    parser.add_argument("--lambda_mmd_grid", default="0.02,0.05,0.1")
    parser.add_argument("--soft_tau_grid", default="0.15,0.25,0.4")

    parser.add_argument("--quick", action="store_true", help="Tiny end-to-end smoke test of every stage.")
    args = parser.parse_args()

    def parse_int_list(s):
        return [int(x) for x in s.split(",") if x.strip()]

    def parse_float_list(s):
        return [float(x) for x in s.split(",") if x.strip()]

    def parse_str_list(s):
        return [x for x in s.split(",") if x.strip()]

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.baseline_seeds = "0"
        args.stage_a_seeds = "0"
        args.stage_b_seeds = "0"
        args.stage_c_seeds = "0"
        args.stage_c_top = 1
        args.factor_grid = "12,20"
        args.lr_grid = "0.003,0.01"
        args.l2_grid = "1e-5"
        args.user_cluster_grid = "4,8"
        args.item_cluster_grid = "4,8"
        args.time_schemes = "tod3,weekday_weekend"
        args.default_user_clusters = 4
        args.default_item_clusters = 4
        args.lambda_kl_grid = "0.01"
        args.lambda_mmd_grid = "0.05"
        args.soft_tau_grid = "0.25"

    return args, parse_int_list, parse_float_list, parse_str_list


def build_base_cfg(args) -> ExperimentConfig:
    return ExperimentConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seeds=[0],
        epochs=args.epochs,
        factors=50,
        batch_size=args.batch_size,
        lr=0.003,
        l2=1e-5,
        lambda_kl=0.01,
        lambda_mmd=0.05,
        lambda_peak=0.0,
        soft_tau=0.25,
        residual_bins=args.residual_bins,
        dot_bins=args.dot_bins,
        user_clusters=args.default_user_clusters,
        item_clusters=args.default_item_clusters,
        top_k=args.top_k,
        max_candidates=None,
        include_oracle=False,
    )


def main():
    args, parse_int_list, parse_float_list, parse_str_list = parse_args()
    base_cfg = build_base_cfg(args)
    ctx = load_context(base_cfg)

    baseline_seeds = parse_int_list(args.baseline_seeds)
    stage_a_seeds = parse_int_list(args.stage_a_seeds)
    stage_b_seeds = parse_int_list(args.stage_b_seeds)
    stage_c_seeds = parse_int_list(args.stage_c_seeds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(args.data_dir),
        "n_train": len(ctx.train),
        "n_val": len(ctx.val),
        "n_test": len(ctx.test),
        "n_users": len(ctx.user_to_idx),
        "n_items": len(ctx.item_to_idx),
        "args": vars(args) | {"data_dir": str(args.data_dir), "out_dir": str(args.out_dir), "baseline_out_dir": str(args.baseline_out_dir)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    stages_to_run = [args.stage] if args.stage != "all" else ["baseline", "a", "b", "c"]

    best_svd_cfg = base_cfg
    winners = None
    stage_b_rows: List[Dict] = []

    if "baseline" not in stages_to_run:
        # Resume path: stages were launched separately (e.g. --stage a after a
        # prior --stage baseline run), so pick up the tuned SVD config from disk
        # instead of silently falling back to the untuned ExperimentConfig
        # defaults (factors=50, lr=0.003, l2=1e-5).
        baseline_summary_path = args.baseline_out_dir / "summary.csv"
        if baseline_summary_path.exists():
            best_row = pd.read_csv(baseline_summary_path).sort_values("mean_test_rmse").iloc[0]
            best_svd_cfg = replace(
                base_cfg, factors=int(best_row["factors"]), lr=float(best_row["lr"]), l2=float(best_row["l2"])
            )
            print(
                f"Loaded tuned SVD config from {baseline_summary_path}: "
                f"factors={best_svd_cfg.factors} lr={best_svd_cfg.lr} l2={best_svd_cfg.l2}"
            )
        else:
            print(
                f"WARNING: {baseline_summary_path} not found; falling back to untuned SVD defaults "
                f"(factors={base_cfg.factors} lr={base_cfg.lr} l2={base_cfg.l2}). Run --stage baseline first."
            )

    if "baseline" in stages_to_run:
        print("\n=== Stage 0: standard SVD tabulation ===")
        summary = run_stage_baseline(
            ctx,
            base_cfg,
            parse_int_list(args.factor_grid),
            parse_float_list(args.lr_grid),
            parse_float_list(args.l2_grid),
            baseline_seeds,
            args.baseline_out_dir,
        )
        best_row = summary.iloc[0]
        best_svd_cfg = replace(base_cfg, factors=int(best_row["factors"]), lr=float(best_row["lr"]), l2=float(best_row["l2"]))
        print(f"Wrote standard SVD tabulation to {args.baseline_out_dir}")
        print(f"Best plain-SVD config: factors={best_svd_cfg.factors} lr={best_svd_cfg.lr} l2={best_svd_cfg.l2}")

    if "a" in stages_to_run:
        print("\n=== Stage A: grouping axis screen ===")
        winners = run_stage_a(
            ctx,
            best_svd_cfg,
            stage_a_seeds,
            args.out_dir,
            parse_int_list(args.user_cluster_grid),
            parse_int_list(args.item_cluster_grid),
            parse_str_list(args.time_schemes),
            args.default_user_clusters,
            args.default_item_clusters,
        )

    if "b" in stages_to_run:
        print("\n=== Stage B: grouping + prior-loss weight search ===")
        if winners is None:
            winners_path = args.out_dir / "stage_a_winners.json"
            if not winners_path.exists():
                raise RuntimeError("Stage B requires Stage A winners; run --stage a first or --stage all.")
            winners = json.loads(winners_path.read_text())
        stage_b_rows = run_stage_b(
            ctx,
            best_svd_cfg,
            stage_b_seeds,
            args.out_dir,
            winners,
            parse_float_list(args.lambda_kl_grid),
            parse_float_list(args.lambda_mmd_grid),
            parse_float_list(args.soft_tau_grid),
        )

    if "c" in stages_to_run:
        print("\n=== Stage C: full-cost confirmation ===")
        if winners is None:
            winners_path = args.out_dir / "stage_a_winners.json"
            if not winners_path.exists():
                raise RuntimeError("Stage C requires Stage A winners; run --stage a first or --stage all.")
            winners = json.loads(winners_path.read_text())
        if not stage_b_rows:
            log_path = args.out_dir / "search_log.csv"
            if not log_path.exists():
                raise RuntimeError("Stage C requires Stage B results; run --stage b first or --stage all.")
            full_log = pd.read_csv(log_path)
            stage_b_rows = full_log[full_log["stage"] == "B"].to_dict("records")
        run_stage_c(
            ctx,
            best_svd_cfg,
            stage_b_rows,
            winners,
            stage_c_seeds,
            args.out_dir / "best_config",
            top_k=args.stage_c_top,
        )

    print(f"\nWrote search outputs under {args.out_dir}")
    print("Key files: search_log.csv, stage_a_winners.json, best_config/top_candidates_winner_screen.csv")


if __name__ == "__main__":
    main()
