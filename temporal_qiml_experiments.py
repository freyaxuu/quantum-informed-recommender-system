#!/usr/bin/env python3
"""Temporal QIML-style prior experiments for MovieLens 100K.

This runner treats each user's ratings as a timestamp-ordered sequence.  It
tests whether priors conditioned on recent user taste state give a more
meaningful recommender analogue of the invariant-measure idea than the earlier
static priors.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from qiml_prior_experiments import (
    GENRE_COLS,
    PriorSpec,
    apply_threshold_bins,
    build_item_genres,
    build_item_type_features,
    compare_distributions,
    fit_quantile_thresholds,
    fit_stat_biases,
    histogram,
    load_movielens,
    normalize_probs,
    prior_loss_and_grad,
    robust_quantile_bins,
    simple_kmeans,
    soft_bins_and_grad,
    temporal_split,
)


@dataclass
class TemporalConfig:
    data_dir: Path
    out_dir: Path
    seeds: List[int]
    epochs: int
    factors: int
    batch_size: int
    lr: float
    l2: float
    lambda_kl: float
    lambda_mmd: float
    lambda_peak: float
    soft_tau: float
    window: int
    residual_bins: int
    delta_bins: int
    state_clusters: int
    item_clusters: int
    gap_bins: int
    top_k: int
    max_candidates: Optional[int]
    windows: Optional[List[int]]


class TemporalMF:
    """MF plus lightweight temporal side features, trained with NumPy Adam."""

    def __init__(self, n_users: int, n_items: int, n_features: int, factors: int, mu: float, seed: int):
        rng = np.random.default_rng(seed)
        scale = 0.05
        self.user_factors = rng.normal(0.0, scale, size=(n_users, factors)).astype(np.float64)
        self.item_factors = rng.normal(0.0, scale, size=(n_items, factors)).astype(np.float64)
        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)
        self.feature_weights = np.zeros(n_features, dtype=np.float64)
        self.mu = float(mu)
        self._adam_t = 0
        self._adam_m = {
            "P": np.zeros_like(self.user_factors),
            "Q": np.zeros_like(self.item_factors),
            "bu": np.zeros_like(self.user_bias),
            "bi": np.zeros_like(self.item_bias),
            "w": np.zeros_like(self.feature_weights),
        }
        self._adam_v = {k: np.zeros_like(v) for k, v in self._adam_m.items()}

    def copy(self) -> "TemporalMF":
        clone = TemporalMF(
            self.user_factors.shape[0],
            self.item_factors.shape[0],
            len(self.feature_weights),
            self.user_factors.shape[1],
            self.mu,
            seed=0,
        )
        clone.user_factors = self.user_factors.copy()
        clone.item_factors = self.item_factors.copy()
        clone.user_bias = self.user_bias.copy()
        clone.item_bias = self.item_bias.copy()
        clone.feature_weights = self.feature_weights.copy()
        clone._adam_t = self._adam_t
        clone._adam_m = {k: v.copy() for k, v in self._adam_m.items()}
        clone._adam_v = {k: v.copy() for k, v in self._adam_v.items()}
        return clone

    def predict(self, users: np.ndarray, items: np.ndarray, x: np.ndarray, return_dot: bool = False):
        dots = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        pred = self.mu + self.user_bias[users] + self.item_bias[items] + dots + x @ self.feature_weights
        if return_dot:
            return pred, dots
        return pred

    def update(
        self,
        users: np.ndarray,
        items: np.ndarray,
        x: np.ndarray,
        grad_pred: np.ndarray,
        lr: float,
        l2: float,
    ) -> None:
        p = self.user_factors[users]
        q = self.item_factors[items]
        grad_p_batch = grad_pred[:, None] * q + l2 * p
        grad_q_batch = grad_pred[:, None] * p + l2 * q
        grad_bu_batch = grad_pred + l2 * self.user_bias[users]
        grad_bi_batch = grad_pred + l2 * self.item_bias[items]
        grad_w = x.T @ grad_pred + l2 * self.feature_weights

        grad_p = np.zeros_like(self.user_factors)
        grad_q = np.zeros_like(self.item_factors)
        grad_bu = np.zeros_like(self.user_bias)
        grad_bi = np.zeros_like(self.item_bias)
        np.add.at(grad_p, users, grad_p_batch)
        np.add.at(grad_q, items, grad_q_batch)
        np.add.at(grad_bu, users, grad_bu_batch)
        np.add.at(grad_bi, items, grad_bi_batch)

        self._adam_step("P", self.user_factors, grad_p, lr)
        self._adam_step("Q", self.item_factors, grad_q, lr)
        self._adam_step("bu", self.user_bias, grad_bu, lr)
        self._adam_step("bi", self.item_bias, grad_bi, lr)
        self._adam_step("w", self.feature_weights, grad_w, lr)

    def _adam_step(self, name: str, param: np.ndarray, grad: np.ndarray, lr: float) -> None:
        if name == "P":
            self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
        self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad * grad)
        m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
        v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


def add_indices(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    user_ids = sorted(train["user_id"].unique())
    item_ids = sorted(train["item_id"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {i: j for j, i in enumerate(item_ids)}

    def map_known(df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["user_id"].isin(user_to_idx) & df["item_id"].isin(item_to_idx)].copy()
        out["user_idx"] = out["user_id"].map(user_to_idx).astype(np.int64)
        out["item_idx"] = out["item_id"].map(item_to_idx).astype(np.int64)
        return out.reset_index(drop=True)

    return map_known(train), map_known(val), map_known(test), user_to_idx, item_to_idx


def add_stat_columns(df: pd.DataFrame, user_bias: Dict[int, float], item_bias: Dict[int, float], mu: float):
    out = df.copy()
    out["bu_stat"] = out["user_id"].map(user_bias).fillna(0.0).astype(float)
    out["bi_stat"] = out["item_id"].map(item_bias).fillna(0.0).astype(float)
    out["residual_stat"] = out["rating"].astype(float) - mu - out["bu_stat"] - out["bi_stat"]
    return out


def kmeans_with_centers(x: np.ndarray, k: int, seed: int, n_iter: int = 60):
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(x), size=min(k, len(x)), replace=False)
    centers = x[init_idx].copy()
    if len(centers) < k:
        pad = np.repeat(centers[-1:], k - len(centers), axis=0)
        centers = np.vstack([centers, pad])
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(n_iter):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            centers[j] = x[mask].mean(axis=0) if mask.any() else x[rng.integers(0, len(x))]
    return labels, centers


def assign_nearest(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist2, axis=1).astype(np.int64)


def build_item_cluster_maps(train: pd.DataFrame, items: pd.DataFrame, item_to_idx: Dict[int, int], cfg: TemporalConfig):
    genre_cols, item_genres, item_genre_matrix = build_item_genres(items, item_to_idx)
    item_x = build_item_type_features(train, len(item_to_idx), item_genre_matrix)
    labels = simple_kmeans(item_x, cfg.item_clusters, seed=42)
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    item_cluster = {int(idx_to_item[idx]): int(label) for idx, label in enumerate(labels)}
    return genre_cols, item_genres, item_genre_matrix, item_cluster


def history_feature(prev: pd.DataFrame, item_genres: Dict[int, List[int]], n_genres: int):
    residuals = prev["residual_stat"].to_numpy(dtype=np.float64)
    ratings = prev["rating"].to_numpy(dtype=np.float64)
    genre_exposure = np.zeros(n_genres, dtype=np.float64)
    genre_lift = np.zeros(n_genres, dtype=np.float64)
    genre_count = np.zeros(n_genres, dtype=np.float64)
    for item_id, residual in zip(prev["item_id"], residuals):
        genres = item_genres.get(int(item_id), [])
        if not genres:
            continue
        w = 1.0 / len(genres)
        for g in genres:
            genre_exposure[g] += w
            genre_lift[g] += w * residual
            genre_count[g] += w
    if genre_exposure.sum() > 0:
        genre_exposure /= genre_exposure.sum()
    genre_lift = np.divide(genre_lift, genre_count, out=np.zeros_like(genre_lift), where=genre_count > 0)
    scalars = np.array(
        [
            residuals.mean(),
            residuals.std() if len(residuals) > 1 else 0.0,
            ratings.mean() - 3.5,
            (ratings >= 4).mean(),
            residuals[-1],
        ],
        dtype=np.float64,
    )
    return np.concatenate([genre_exposure, genre_lift, scalars])


def target_genre_score(prev: pd.DataFrame, target_item_id: int, item_genres: Dict[int, List[int]]):
    target_genres = item_genres.get(int(target_item_id), [])
    if not target_genres:
        return 0.0
    values = []
    for g in target_genres:
        vals = []
        for row in prev.itertuples(index=False):
            if g in item_genres.get(int(row.item_id), []):
                vals.append(float(row.residual_stat))
        if vals:
            values.append(float(np.mean(vals)))
    return float(np.mean(values)) if values else 0.0


def build_sequence_rows(
    ratings_all: pd.DataFrame,
    train_raw: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
    item_genres: Dict[int, List[int]],
    item_cluster: Dict[int, int],
    n_genres: int,
    cfg: TemporalConfig,
):
    rows = []
    hist_features = []
    curr_features = []
    for _, user_df in ratings_all.sort_values(["user_id", "timestamp"]).groupby("user_id", sort=False):
        user_df = user_df.reset_index(drop=True)
        n = len(user_df)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, int(n * 0.1))
        for pos in range(cfg.window, n):
            row = user_df.iloc[pos]
            if int(row.user_id) not in user_to_idx or int(row.item_id) not in item_to_idx:
                continue
            split = "train" if pos < n_train else "val" if pos < n_train + n_val else "test"
            prev = user_df.iloc[pos - cfg.window : pos]
            curr_window = user_df.iloc[pos - cfg.window + 1 : pos + 1]
            h_prev = history_feature(prev, item_genres, n_genres)
            h_curr = history_feature(curr_window, item_genres, n_genres)
            gap_seconds = float(row.timestamp - user_df.iloc[pos - 1].timestamp)
            rows.append(
                {
                    "split": split,
                    "user_id": int(row.user_id),
                    "item_id": int(row.item_id),
                    "rating": float(row.rating),
                    "timestamp": int(row.timestamp),
                    "user_idx": int(user_to_idx[int(row.user_id)]),
                    "item_idx": int(item_to_idx[int(row.item_id)]),
                    "position": pos,
                    "seq_pos_norm": pos / max(1, n - 1),
                    "bu_stat": float(row.bu_stat),
                    "bi_stat": float(row.bi_stat),
                    "residual_stat": float(row.residual_stat),
                    "prev_residual": float(prev.iloc[-1].residual_stat),
                    "delta_residual_stat": float(row.residual_stat - prev.iloc[-1].residual_stat),
                    "recent_mean_residual": float(prev["residual_stat"].mean()),
                    "recent_std_residual": float(prev["residual_stat"].std(ddof=0)),
                    "recent_mean_rating_centered": float(prev["rating"].mean() - 3.5),
                    "recent_positive_rate": float((prev["rating"] >= 4).mean()),
                    "recent_genre_score": target_genre_score(prev, int(row.item_id), item_genres),
                    "gap_seconds": gap_seconds,
                    "log_gap": np.log1p(max(0.0, gap_seconds)),
                    "item_cluster": int(item_cluster.get(int(row.item_id), 0)),
                }
            )
            hist_features.append(h_prev)
            curr_features.append(h_curr)
    seq = pd.DataFrame(rows)
    hist_x = np.vstack(hist_features)
    curr_x = np.vstack(curr_features)

    # Fit taste states only on train previous-window features.
    train_mask = seq["split"].to_numpy() == "train"
    feat_mean = hist_x[train_mask].mean(axis=0)
    feat_std = hist_x[train_mask].std(axis=0)
    feat_std[feat_std == 0] = 1.0
    hist_z = (hist_x - feat_mean) / feat_std
    curr_z = (curr_x - feat_mean) / feat_std
    _, centers = kmeans_with_centers(hist_z[train_mask], cfg.state_clusters, seed=42)
    seq["prev_state"] = assign_nearest(hist_z, centers)
    seq["curr_state"] = assign_nearest(curr_z, centers)

    train_gap = seq.loc[train_mask, "log_gap"]
    gap_thresholds = fit_quantile_thresholds(train_gap, cfg.gap_bins)
    seq["gap_bucket"] = apply_threshold_bins(seq["log_gap"], gap_thresholds).to_numpy()

    prev_edges, _ = robust_quantile_bins(seq.loc[train_mask, "prev_residual"].to_numpy(), cfg.residual_bins)
    seq["prev_residual_bin"] = np.searchsorted(prev_edges[1:-1], seq["prev_residual"].to_numpy(), side="right")
    return seq, centers


def standardize_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [
        "seq_pos_norm",
        "recent_mean_residual",
        "recent_std_residual",
        "recent_mean_rating_centered",
        "recent_positive_rate",
        "recent_genre_score",
        "prev_residual",
        "log_gap",
    ]
    mean = train[feature_cols].mean()
    std = train[feature_cols].std(ddof=0).replace(0, 1.0)
    for df in (train, val, test):
        for col in feature_cols:
            df[f"x_{col}"] = (df[col] - mean[col]) / std[col]
    return [f"x_{col}" for col in feature_cols]


def attach_group_column(df_list: Iterable[pd.DataFrame], name: str, columns: List[str], cardinalities: List[int]):
    multipliers = []
    m = 1
    for card in reversed(cardinalities[1:]):
        m *= card
        multipliers.insert(0, m)
    multipliers.append(1)
    col = f"{name}_group_idx"
    for df in df_list:
        idx = np.zeros(len(df), dtype=np.int64)
        for c, mult in zip(columns, multipliers):
            idx += df[c].to_numpy(dtype=np.int64) * mult
        df[col] = idx
    return col, int(np.prod(cardinalities))


def grouped_histogram(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n_groups: int,
    centers: np.ndarray,
    edges: np.ndarray,
    fallback: np.ndarray,
):
    probs = np.zeros((n_groups, len(centers)), dtype=np.float64)
    for g in range(n_groups):
        values = df.loc[df[group_col] == g, value_col].to_numpy()
        probs[g] = fallback if len(values) == 0 else histogram(values, centers, edges)
    return normalize_probs(probs)


def make_temporal_priors(cfg: TemporalConfig, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    residual_edges, residual_centers = robust_quantile_bins(train["residual_stat"].to_numpy(), cfg.residual_bins)
    delta_edges, delta_centers = robust_quantile_bins(train["delta_residual_stat"].to_numpy(), cfg.delta_bins)
    global_resid = histogram(train["residual_stat"].to_numpy(), residual_centers, residual_edges)
    global_delta = histogram(train["delta_residual_stat"].to_numpy(), delta_centers, delta_edges)

    priors = []
    priors.append(PriorSpec("temporal_global_residual", "residual", residual_centers, global_resid, "batch"))

    col, n_groups = attach_group_column(
        (train, val, test),
        "temporal_next_residual",
        ["prev_state", "item_cluster"],
        [cfg.state_clusters, cfg.item_clusters],
    )
    priors.append(
        PriorSpec(
            "temporal_next_residual",
            "residual",
            residual_centers,
            grouped_histogram(train, col, "residual_stat", n_groups, residual_centers, residual_edges, global_resid),
            "sample",
            ["prev_state", "item_cluster"],
            "P(E_t | S_{t-1}, item_type_t)",
        )
    )

    col, n_groups = attach_group_column(
        (train, val, test),
        "temporal_delta_residual",
        ["prev_residual_bin", "prev_state", "item_cluster"],
        [cfg.residual_bins, cfg.state_clusters, cfg.item_clusters],
    )
    priors.append(
        PriorSpec(
            "temporal_delta_residual",
            "delta_residual",
            delta_centers,
            grouped_histogram(train, col, "delta_residual_stat", n_groups, delta_centers, delta_edges, global_delta),
            "sample",
            ["prev_residual_bin", "prev_state", "item_cluster"],
            "P(delta E_t | E_{t-1 bin}, S_{t-1}, item_type_t)",
        )
    )

    col, n_groups = attach_group_column(
        (train, val, test),
        "temporal_gap_residual",
        ["gap_bucket", "prev_state", "item_cluster"],
        [cfg.gap_bins, cfg.state_clusters, cfg.item_clusters],
    )
    priors.append(
        PriorSpec(
            "temporal_gap_residual",
            "residual",
            residual_centers,
            grouped_histogram(train, col, "residual_stat", n_groups, residual_centers, residual_edges, global_resid),
            "sample",
            ["gap_bucket", "prev_state", "item_cluster"],
            "P(E_t | gap_bucket, S_{t-1}, item_type_t)",
        )
    )

    for prior in priors:
        for df in (train, val, test):
            if prior.is_grouped:
                rng = np.random.default_rng(1000 + len(prior.name) + len(df))
                df[f"{prior.name}_time_shuffled_group_idx"] = rng.permutation(df[prior.group_col].to_numpy())
    if cfg.max_candidates is not None:
        priors = priors[: cfg.max_candidates]
    return priors


def train_temporal_model(
    model: TemporalMF,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: List[str],
    cfg: TemporalConfig,
    prior: Optional[PriorSpec],
    seed: int,
):
    rng = np.random.default_rng(seed)
    users = train["user_idx"].to_numpy(np.int64)
    items = train["item_idx"].to_numpy(np.int64)
    ratings = train["rating"].to_numpy(np.float64)
    bu_stat = train["bu_stat"].to_numpy(np.float64)
    bi_stat = train["bi_stat"].to_numpy(np.float64)
    prev_residual = train["prev_residual"].to_numpy(np.float64)
    x = train[feature_cols].to_numpy(np.float64)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(train))
        mse_vals, prior_vals, kl_vals, mmd_vals = [], [], [], []
        for start in range(0, len(train), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            pred, _ = model.predict(users[idx], items[idx], x[idx], return_dot=True)
            err = pred - ratings[idx]
            mse = float(np.mean(err * err))
            grad_pred = 2.0 * err / len(idx)
            ploss = 0.0
            stats = {"prior_kl": 0.0, "prior_mmd": 0.0}
            if prior is not None:
                batch_df = train.iloc[idx]
                if prior.value_kind == "residual":
                    values = pred - model.mu - bu_stat[idx] - bi_stat[idx]
                elif prior.value_kind == "delta_residual":
                    values = pred - model.mu - bu_stat[idx] - bi_stat[idx] - prev_residual[idx]
                else:
                    raise ValueError(f"Unsupported temporal prior kind: {prior.value_kind}")
                loss, grad_value, stats = prior_loss_and_grad(values, prior, batch_df, cfg)  # type: ignore[arg-type]
                grad_pred += grad_value
                ploss = loss
            model.update(users[idx], items[idx], x[idx], grad_pred, cfg.lr, cfg.l2)
            mse_vals.append(mse)
            prior_vals.append(ploss)
            kl_vals.append(stats.get("prior_kl", 0.0))
            mmd_vals.append(stats.get("prior_mmd", 0.0))
        val_metrics = rating_metrics(model, val, feature_cols)
        history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(mse_vals)),
                "train_prior_loss": float(np.mean(prior_vals)),
                "train_prior_kl": float(np.mean(kl_vals)),
                "train_prior_mmd": float(np.mean(mmd_vals)),
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
            }
        )
    return model, pd.DataFrame(history)


def rating_metrics(model: TemporalMF, df: pd.DataFrame, feature_cols: List[str]):
    pred = model.predict(
        df["user_idx"].to_numpy(np.int64),
        df["item_idx"].to_numpy(np.int64),
        df[feature_cols].to_numpy(np.float64),
    )
    truth = df["rating"].to_numpy(np.float64)
    clipped = np.clip(pred, 1.0, 5.0)
    return {
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "clipped_rmse": float(np.sqrt(np.mean((clipped - truth) ** 2))),
        "clipped_mae": float(np.mean(np.abs(clipped - truth))),
    }


def temporal_distribution_metrics(
    model: TemporalMF,
    df: pd.DataFrame,
    feature_cols: List[str],
    prior: PriorSpec,
    cfg: TemporalConfig,
):
    pred = model.predict(
        df["user_idx"].to_numpy(np.int64),
        df["item_idx"].to_numpy(np.int64),
        df[feature_cols].to_numpy(np.float64),
    )
    if prior.value_kind == "residual":
        values = pred - model.mu - df["bu_stat"].to_numpy() - df["bi_stat"].to_numpy()
    elif prior.value_kind == "delta_residual":
        values = pred - model.mu - df["bu_stat"].to_numpy() - df["bi_stat"].to_numpy() - df["prev_residual"].to_numpy()
    else:
        raise ValueError(f"Unsupported temporal prior kind: {prior.value_kind}")
    q, _ = soft_bins_and_grad(values, prior.centers, cfg.soft_tau)
    pred_dist = normalize_probs(q.mean(axis=0))
    target = normalize_probs(prior.target_for_rows(df).mean(axis=0)) if prior.is_grouped else prior.probs
    return compare_distributions(pred_dist, target, prior.centers)


def ranking_metrics(model: TemporalMF, train: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str], n_items: int, k: int):
    seen = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    positives = test[test["rating"] >= 4].groupby("user_idx")["item_idx"].apply(set).to_dict()
    if not positives:
        return {"recall_at_k": np.nan, "ndcg_at_k": np.nan, "map_at_k": np.nan}
    all_items = np.arange(n_items, dtype=np.int64)
    recalls, ndcgs, maps = [], [], []
    by_user_row = test.sort_values("timestamp").groupby("user_idx").first()
    for user, pos_items in positives.items():
        candidates = np.array([i for i in all_items if i not in seen.get(user, set())], dtype=np.int64)
        if len(candidates) == 0 or user not in by_user_row.index:
            continue
        row = by_user_row.loc[user]
        users = np.full(len(candidates), int(user), dtype=np.int64)
        x = np.repeat(row[feature_cols].to_numpy(dtype=np.float64)[None, :], len(candidates), axis=0)
        scores = model.predict(users, candidates, x)
        top = candidates[np.argsort(scores)[-k:][::-1]]
        hits = np.array([1.0 if int(item) in pos_items else 0.0 for item in top])
        recalls.append(float(hits.sum() / max(1, len(pos_items))))
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        ideal = float(np.sum(discounts[: min(k, len(pos_items))]))
        ndcgs.append(float(np.sum(hits * discounts) / ideal) if ideal > 0 else 0.0)
        precisions = []
        h = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                h += 1.0
                precisions.append(h / rank)
        maps.append(float(np.mean(precisions)) if precisions else 0.0)
    return {"recall_at_k": float(np.mean(recalls)), "ndcg_at_k": float(np.mean(ndcgs)), "map_at_k": float(np.mean(maps))}


def residual_autocorr_error(model: TemporalMF, df: pd.DataFrame, feature_cols: List[str], lags=(1, 2, 3, 5, 10)):
    pred = model.predict(
        df["user_idx"].to_numpy(np.int64),
        df["item_idx"].to_numpy(np.int64),
        df[feature_cols].to_numpy(np.float64),
    )
    tmp = df[["user_id", "timestamp", "residual_stat", "bu_stat", "bi_stat"]].copy()
    tmp["pred_residual"] = pred - model.mu - tmp["bu_stat"] - tmp["bi_stat"]

    true_vals, pred_vals = [], []
    for lag in lags:
        true_pairs, pred_pairs = [], []
        for _, user_df in tmp.sort_values("timestamp").groupby("user_id"):
            if len(user_df) <= lag:
                continue
            a = user_df["residual_stat"].to_numpy()
            b = user_df["pred_residual"].to_numpy()
            true_pairs.append(corr_at_lag(a, lag))
            pred_pairs.append(corr_at_lag(b, lag))
        true_vals.append(np.nanmean(true_pairs) if true_pairs else np.nan)
        pred_vals.append(np.nanmean(pred_pairs) if pred_pairs else np.nan)
    true_arr = np.nan_to_num(np.array(true_vals, dtype=np.float64))
    pred_arr = np.nan_to_num(np.array(pred_vals, dtype=np.float64))
    denom = np.linalg.norm(true_arr)
    return float(np.linalg.norm(true_arr - pred_arr) / denom) if denom > 0 else np.nan


def corr_at_lag(values: np.ndarray, lag: int):
    x = values[:-lag]
    y = values[lag:]
    if len(x) < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else np.nan


def collect_metrics(variant, candidate, seed, model, train, val, test, feature_cols, prior, cfg, n_items):
    row = {"seed": seed, "candidate": candidate, "variant": variant}
    row.update({f"val_{k}": v for k, v in rating_metrics(model, val, feature_cols).items()})
    row.update({f"test_{k}": v for k, v in rating_metrics(model, test, feature_cols).items()})
    row.update(ranking_metrics(model, train, test, feature_cols, n_items, cfg.top_k))
    row["residual_autocorr_error"] = residual_autocorr_error(model, test, feature_cols)
    if prior is not None:
        dist = temporal_distribution_metrics(model, test, feature_cols, prior, cfg)
        row.update({f"dist_{k}": v for k, v in dist.items()})
    return row


def subgroup_metrics(model, train, test, feature_cols):
    user_count = train.groupby("user_id").size()
    item_count = train.groupby("item_id").size()
    df = test.copy()
    df["cold_user"] = df["user_id"].map(user_count).fillna(0) <= 20
    df["long_tail_item"] = df["item_id"].map(item_count).fillna(0) <= 20
    groups = {
        "all": np.ones(len(df), dtype=bool),
        "cold_user": df["cold_user"].to_numpy(bool),
        "warm_user": ~df["cold_user"].to_numpy(bool),
        "long_tail_item": df["long_tail_item"].to_numpy(bool),
        "popular_item": ~df["long_tail_item"].to_numpy(bool),
    }
    rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        rows.append({"subgroup": name, "n": int(mask.sum()), **rating_metrics(model, df.loc[mask], feature_cols)})
    return pd.DataFrame(rows)


def save_prior(spec: PriorSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{spec.name}.npz",
        name=np.array(spec.name),
        value_kind=np.array(spec.value_kind),
        mode=np.array(spec.mode),
        centers=spec.centers,
        probs=spec.probs,
        description=np.array(spec.description),
        group_keys=np.array(spec.group_keys),
    )


def state_occupancy(df: pd.DataFrame, cfg: TemporalConfig, state_col: str = "prev_state"):
    counts = df[state_col].value_counts().reindex(range(cfg.state_clusters), fill_value=0).to_numpy()
    return normalize_probs(counts)


def state_transition_matrix(df: pd.DataFrame, cfg: TemporalConfig):
    counts = np.zeros((cfg.state_clusters, cfg.state_clusters), dtype=np.float64)
    for row in df.itertuples(index=False):
        counts[int(row.prev_state), int(row.curr_state)] += 1.0
    fallback = np.ones(cfg.state_clusters, dtype=np.float64) / cfg.state_clusters
    probs = np.zeros_like(counts)
    for state in range(cfg.state_clusters):
        probs[state] = fallback if counts[state].sum() == 0 else normalize_probs(counts[state])
    return probs, counts


def mean_rowwise_transition_kl(p: np.ndarray, q: np.ndarray, weights: np.ndarray):
    weights = normalize_probs(weights)
    kls = []
    for i in range(p.shape[0]):
        kls.append(compare_distributions(p[i], q[i], np.arange(p.shape[1]))["kl"])
    return float(np.sum(np.asarray(kls) * weights))


def empirical_autocorr(df: pd.DataFrame, lags=(1, 2, 3, 5, 10)):
    values = []
    for lag in lags:
        per_user = []
        for _, user_df in df.sort_values("timestamp").groupby("user_id"):
            if len(user_df) <= lag:
                continue
            per_user.append(corr_at_lag(user_df["residual_stat"].to_numpy(dtype=np.float64), lag))
        values.append(np.nanmean(per_user) if per_user else np.nan)
    return np.asarray(values, dtype=np.float64)


def temporal_diagnostics(train, val, test, cfg, out_dir):
    rows = []
    table_dir = out_dir / "diagnostic_priors"
    table_dir.mkdir(parents=True, exist_ok=True)

    train_occ = state_occupancy(train, cfg)
    train_curr_occ = state_occupancy(train, cfg, "curr_state")
    train_trans, train_trans_counts = state_transition_matrix(train, cfg)
    train_trans_weights = train["prev_state"].value_counts().reindex(range(cfg.state_clusters), fill_value=0).to_numpy()
    train_autocorr = empirical_autocorr(train)

    np.savez_compressed(
        table_dir / "state_occupancy.npz",
        prev_state_probs=train_occ,
        curr_state_probs=train_curr_occ,
        description=np.array("Train empirical state occupancy prior"),
    )
    np.savez_compressed(
        table_dir / "state_transition.npz",
        probs=train_trans,
        counts=train_trans_counts,
        description=np.array("Train empirical P(S_t | S_{t-1}) transition prior"),
    )
    np.savez_compressed(
        table_dir / "residual_autocorr.npz",
        lags=np.array([1, 2, 3, 5, 10], dtype=np.int64),
        autocorr=train_autocorr,
        description=np.array("Train empirical residual autocorrelation curve"),
    )

    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        occ = state_occupancy(df, cfg)
        rows.append({"diagnostic": "state_occupancy", "split": split_name, "value": np.nan, "json": json.dumps(occ.tolist())})
    for split_name, df in [("val", val), ("test", test)]:
        occ = state_occupancy(df, cfg)
        rows.append({"diagnostic": "state_occupancy_kl", "split": split_name, "value": compare_distributions(occ, train_occ, np.arange(cfg.state_clusters))["kl"], "json": ""})
        trans, _ = state_transition_matrix(df, cfg)
        rows.append(
            {
                "diagnostic": "state_transition_flat_kl",
                "split": split_name,
                "value": compare_distributions(trans.ravel(), train_trans.ravel(), np.arange(trans.size))["kl"],
                "json": "",
            }
        )
        rows.append(
            {
                "diagnostic": "state_transition_rowwise_kl",
                "split": split_name,
                "value": mean_rowwise_transition_kl(trans, train_trans, train_trans_weights),
                "json": "",
            }
        )
        autocorr = empirical_autocorr(df)
        denom = np.linalg.norm(np.nan_to_num(train_autocorr))
        error = np.linalg.norm(np.nan_to_num(autocorr) - np.nan_to_num(train_autocorr)) / denom if denom > 0 else np.nan
        rows.append({"diagnostic": "residual_autocorr_error", "split": split_name, "value": float(error), "json": json.dumps(autocorr.tolist())})
    pd.DataFrame(rows).to_csv(out_dir / "temporal_diagnostics.csv", index=False)


def summarize(out_dir: Path):
    metrics = pd.read_csv(out_dir / "metrics.csv")
    base = metrics[metrics["variant"] == "baseline"][["seed", "test_rmse", "dist_kl"]].rename(columns={"test_rmse": "baseline_rmse"})
    merged = metrics.merge(base[["seed", "baseline_rmse"]], on="seed", how="left")
    merged["rmse_gain_vs_baseline"] = merged["baseline_rmse"] - merged["test_rmse"]
    summary = (
        merged[merged["variant"] != "baseline"]
        .groupby(["candidate", "variant"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_rmse_gain=("rmse_gain_vs_baseline", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_ndcg_at_k=("ndcg_at_k", "mean"),
            mean_dist_kl=("dist_kl", "mean"),
            mean_dist_js=("dist_js", "mean"),
            mean_autocorr_error=("residual_autocorr_error", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    real = summary[summary["variant"] == "real_temporal_prior"][["candidate", "mean_rmse_gain", "mean_dist_kl", "mean_autocorr_error"]]
    shuf = summary[summary["variant"] == "shuffled_user_prior"][["candidate", "mean_rmse_gain", "mean_dist_kl", "mean_autocorr_error"]].rename(columns={"mean_rmse_gain": "shuffled_rmse_gain", "mean_dist_kl": "shuffled_dist_kl", "mean_autocorr_error": "shuffled_autocorr_error"})
    time_shuf = summary[summary["variant"] == "shuffled_time_prior"][["candidate", "mean_rmse_gain", "mean_dist_kl"]].rename(columns={"mean_rmse_gain": "time_shuffled_rmse_gain", "mean_dist_kl": "time_shuffled_dist_kl"})
    screen = real.merge(shuf, on="candidate", how="left").merge(time_shuf, on="candidate", how="left")
    if not screen.empty:
        screen["beats_user_shuffle_rmse"] = screen["mean_rmse_gain"] > screen["shuffled_rmse_gain"]
        screen["beats_time_shuffle_rmse"] = screen["mean_rmse_gain"] > screen["time_shuffled_rmse_gain"]
        screen["beats_user_shuffle_dist"] = screen["mean_dist_kl"] < screen["shuffled_dist_kl"]
        screen["beats_time_shuffle_dist"] = screen["mean_dist_kl"] < screen["time_shuffled_dist_kl"]
        screen["meaningful_temporal_candidate"] = (
            screen["beats_user_shuffle_rmse"]
            & screen["beats_time_shuffle_rmse"]
            & screen["beats_user_shuffle_dist"]
            & screen["beats_time_shuffle_dist"]
        )
        screen.sort_values(["meaningful_temporal_candidate", "mean_rmse_gain"], ascending=[False, False]).to_csv(out_dir / "winner_screen.csv", index=False)


def run(cfg: TemporalConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ratings, items = load_movielens(cfg.data_dir)
    train_raw, val_raw, test_raw = temporal_split(ratings)
    train_raw, val_raw, test_raw, user_to_idx, item_to_idx = add_indices(train_raw, val_raw, test_raw)
    mu = float(train_raw["rating"].mean())
    user_bias, item_bias = fit_stat_biases(train_raw, mu)
    ratings_all = add_stat_columns(ratings, user_bias, item_bias, mu)
    train_raw = add_stat_columns(train_raw, user_bias, item_bias, mu)
    genre_cols, item_genres, item_genre_matrix, item_cluster = build_item_cluster_maps(train_raw, items, item_to_idx, cfg)
    seq, state_centers = build_sequence_rows(ratings_all, train_raw, user_to_idx, item_to_idx, item_genres, item_cluster, len(genre_cols), cfg)
    train = seq[seq["split"] == "train"].copy().reset_index(drop=True)
    val = seq[seq["split"] == "val"].copy().reset_index(drop=True)
    test = seq[seq["split"] == "test"].copy().reset_index(drop=True)
    feature_cols = standardize_features(train, val, test)
    priors = make_temporal_priors(cfg, train, val, test)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_users": len(user_to_idx),
        "n_items": len(item_to_idx),
        "window": cfg.window,
        "feature_cols": feature_cols,
        "config": vars(cfg) | {"data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    temporal_diagnostics(train, val, test, cfg, cfg.out_dir)

    metrics_rows, history_rows, subgroup_rows, prior_rows = [], [], [], []
    n_features = len(feature_cols)
    for seed in cfg.seeds:
        print(f"\n=== seed {seed} ===")
        initial = TemporalMF(len(user_to_idx), len(item_to_idx), n_features, cfg.factors, mu, seed)
        baseline = initial.copy()
        baseline, hist = train_temporal_model(baseline, train, val, feature_cols, cfg, None, seed)
        metrics_rows.append(collect_metrics("baseline", "none", seed, baseline, train, val, test, feature_cols, None, cfg, len(item_to_idx)))
        for _, row in hist.iterrows():
            history_rows.append({"seed": seed, "candidate": "none", "variant": "baseline", **row.to_dict()})
        sg = subgroup_metrics(baseline, train, test, feature_cols)
        sg["seed"] = seed
        sg["candidate"] = "none"
        sg["variant"] = "baseline"
        subgroup_rows.extend(sg.to_dict("records"))

        for spec in priors:
            print(f"candidate={spec.name}")
            save_prior(spec, cfg.out_dir / "priors")
            prior_rows.append(
                {
                    "candidate": spec.name,
                    "value_kind": spec.value_kind,
                    "n_groups": spec.probs.shape[0] if spec.is_grouped else 1,
                    "n_bins": len(spec.centers),
                    "description": spec.description,
                }
            )
            variants = [
                ("real_temporal_prior", spec),
                ("shuffled_user_prior", spec.shuffled(np.random.default_rng(seed + 101))),
                (
                    "shuffled_time_prior",
                    PriorSpec(
                        f"{spec.name}_time_shuffled",
                        spec.value_kind,
                        spec.centers,
                        spec.probs,
                        spec.mode,
                        spec.group_keys,
                        f"Time-shuffled groups for {spec.name}",
                        f"{spec.name}_time_shuffled_group_idx" if spec.is_grouped else spec.group_col,
                    ),
                ),
                ("uniform_prior", spec.uniform()),
                ("random_prior", spec.random(np.random.default_rng(seed + 202))),
            ]
            for variant, prior in variants:
                model = initial.copy()
                model, hist = train_temporal_model(model, train, val, feature_cols, cfg, prior, seed)
                metrics_rows.append(collect_metrics(variant, spec.name, seed, model, train, val, test, feature_cols, prior, cfg, len(item_to_idx)))
                for _, row in hist.iterrows():
                    history_rows.append({"seed": seed, "candidate": spec.name, "variant": variant, **row.to_dict()})
                sg = subgroup_metrics(model, train, test, feature_cols)
                sg["seed"] = seed
                sg["candidate"] = spec.name
                sg["variant"] = variant
                subgroup_rows.extend(sg.to_dict("records"))

        pd.DataFrame(metrics_rows).to_csv(cfg.out_dir / "metrics.csv", index=False)
        pd.DataFrame(history_rows).to_csv(cfg.out_dir / "history.csv", index=False)
        pd.DataFrame(subgroup_rows).to_csv(cfg.out_dir / "subgroups.csv", index=False)
        pd.DataFrame(prior_rows).drop_duplicates("candidate").to_csv(cfg.out_dir / "prior_catalog.csv", index=False)
        summarize(cfg.out_dir)
    return cfg.out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal QIML-style prior experiments.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/temporal_qiml_prior_experiments"))
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
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--windows", default=None, help="Optional comma-separated sweep, for example 5,10,20.")
    parser.add_argument("--residual_bins", type=int, default=16)
    parser.add_argument("--delta_bins", type=int, default=16)
    parser.add_argument("--state_clusters", type=int, default=12)
    parser.add_argument("--item_clusters", type=int, default=16)
    parser.add_argument("--gap_bins", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        seeds = [0]
        args.epochs = min(args.epochs, 2)
        args.factors = min(args.factors, 12)
        args.max_candidates = 2 if args.max_candidates is None else args.max_candidates
    return TemporalConfig(
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
        window=args.window,
        residual_bins=args.residual_bins,
        delta_bins=args.delta_bins,
        state_clusters=args.state_clusters,
        item_clusters=args.item_clusters,
        gap_bins=args.gap_bins,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        windows=[int(w) for w in args.windows.split(",") if w.strip()] if args.windows else None,
    )


def main():
    cfg = parse_args()
    if cfg.windows:
        summary_rows = []
        for window in cfg.windows:
            window_cfg = replace(cfg, window=window, windows=None, out_dir=cfg.out_dir / f"window_{window}")
            out_dir = run(window_cfg)
            summary_path = out_dir / "winner_screen.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path)
                df.insert(0, "window", window)
                summary_rows.extend(df.to_dict("records"))
        if summary_rows:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(summary_rows).to_csv(cfg.out_dir / "window_sweep_winner_screen.csv", index=False)
        print(f"\nWrote temporal QIML window sweep outputs under {cfg.out_dir}")
        print("Key files: window_*/summary.csv, window_*/winner_screen.csv, window_sweep_winner_screen.csv")
    else:
        out_dir = run(cfg)
        print(f"\nWrote temporal QIML prior outputs to {out_dir}")
        print("Key files: summary.csv, winner_screen.csv, temporal_diagnostics.csv, prior_catalog.csv")


if __name__ == "__main__":
    main()
