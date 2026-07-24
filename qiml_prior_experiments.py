#!/usr/bin/env python3
"""Prior-first QIML-style experiments for MovieLens 100K.

This script is intentionally notebook-independent.  It runs a paired
baseline-vs-prior campaign using only numpy/pandas so the first research step
does not depend on quantum tooling.  Candidate priors that survive the
shuffled/null controls can later be exported and fitted with a QCBM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


GENRE_COLS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class ExperimentConfig:
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
    residual_bins: int
    dot_bins: int
    user_clusters: int
    item_clusters: int
    top_k: int
    max_candidates: Optional[int]
    include_oracle: bool


class MatrixFactorization:
    """Small explicit-feedback MF model trained with NumPy Adam."""

    def __init__(self, n_users: int, n_items: int, factors: int, mu: float, seed: int):
        rng = np.random.default_rng(seed)
        scale = 0.05
        self.user_factors = rng.normal(0.0, scale, size=(n_users, factors)).astype(np.float64)
        self.item_factors = rng.normal(0.0, scale, size=(n_items, factors)).astype(np.float64)
        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)
        self.mu = float(mu)

        self._adam_m = {
            "P": np.zeros_like(self.user_factors),
            "Q": np.zeros_like(self.item_factors),
            "bu": np.zeros_like(self.user_bias),
            "bi": np.zeros_like(self.item_bias),
        }
        self._adam_v = {
            "P": np.zeros_like(self.user_factors),
            "Q": np.zeros_like(self.item_factors),
            "bu": np.zeros_like(self.user_bias),
            "bi": np.zeros_like(self.item_bias),
        }
        self._adam_t = 0

    def copy(self) -> "MatrixFactorization":
        clone = MatrixFactorization(
            self.user_factors.shape[0],
            self.item_factors.shape[0],
            self.user_factors.shape[1],
            self.mu,
            seed=0,
        )
        clone.user_factors = self.user_factors.copy()
        clone.item_factors = self.item_factors.copy()
        clone.user_bias = self.user_bias.copy()
        clone.item_bias = self.item_bias.copy()
        clone._adam_m = {k: v.copy() for k, v in self._adam_m.items()}
        clone._adam_v = {k: v.copy() for k, v in self._adam_v.items()}
        clone._adam_t = self._adam_t
        return clone

    def predict(self, users: np.ndarray, items: np.ndarray, return_dot: bool = False):
        dots = np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        pred = self.mu + self.user_bias[users] + self.item_bias[items] + dots
        if return_dot:
            return pred, dots
        return pred

    def update(
        self,
        users: np.ndarray,
        items: np.ndarray,
        grad_pred: np.ndarray,
        grad_dot: Optional[np.ndarray],
        lr: float,
        l2: float,
    ) -> None:
        """Apply one Adam update from gradients wrt predictions and dot terms."""

        if grad_dot is None:
            grad_dot = np.zeros_like(grad_pred)
        total_dot_grad = grad_pred + grad_dot

        p = self.user_factors[users]
        q = self.item_factors[items]

        grad_p_batch = total_dot_grad[:, None] * q + l2 * p
        grad_q_batch = total_dot_grad[:, None] * p + l2 * q
        grad_bu_batch = grad_pred + l2 * self.user_bias[users]
        grad_bi_batch = grad_pred + l2 * self.item_bias[items]

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

    def _adam_step(self, name: str, param: np.ndarray, grad: np.ndarray, lr: float) -> None:
        if name == "P":
            self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
        self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad * grad)
        m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
        v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)


class PriorSpec:
    """Distributional prior over ratings, residuals, or dot products."""

    def __init__(
        self,
        name: str,
        value_kind: str,
        centers: np.ndarray,
        probs: np.ndarray,
        mode: str,
        group_keys: Optional[List[str]] = None,
        description: str = "",
        group_col: Optional[str] = None,
    ):
        self.name = name
        self.value_kind = value_kind
        self.centers = centers.astype(np.float64)
        self.probs = normalize_probs(probs.astype(np.float64))
        self.mode = mode
        self.group_keys = group_keys or []
        self.description = description
        self.group_col = group_col or f"{name}_group_idx"

    @property
    def is_grouped(self) -> bool:
        return self.probs.ndim == 2

    def target_for_rows(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_grouped:
            return np.repeat(self.probs[None, :], len(df), axis=0)
        group_idx = df[self.group_col].to_numpy(dtype=np.int64)
        return self.probs[group_idx]

    def shuffled(self, rng: np.random.Generator) -> "PriorSpec":
        if not self.is_grouped:
            shuffled = self.probs.copy()
            rng.shuffle(shuffled)
            return PriorSpec(
                f"{self.name}_shuffled",
                self.value_kind,
                self.centers,
                shuffled,
                self.mode,
                self.group_keys,
                f"Shuffled bins for {self.name}",
                self.group_col,
            )
        shuffled = self.probs.copy()
        rng.shuffle(shuffled, axis=0)
        return PriorSpec(
            f"{self.name}_shuffled",
            self.value_kind,
            self.centers,
            shuffled,
            self.mode,
            self.group_keys,
            f"Shuffled group assignments for {self.name}",
            self.group_col,
        )

    def uniform(self) -> "PriorSpec":
        if self.is_grouped:
            probs = np.ones_like(self.probs) / self.probs.shape[1]
        else:
            probs = np.ones_like(self.probs) / len(self.probs)
        return PriorSpec(
            f"{self.name}_uniform",
            self.value_kind,
            self.centers,
            probs,
            self.mode,
            self.group_keys,
            f"Uniform null for {self.name}",
            self.group_col,
        )

    def random(self, rng: np.random.Generator) -> "PriorSpec":
        probs = rng.random(self.probs.shape) + 1e-3
        return PriorSpec(
            f"{self.name}_random",
            self.value_kind,
            self.centers,
            probs,
            self.mode,
            self.group_keys,
            f"Random null for {self.name}",
            self.group_col,
        )


def normalize_probs(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, eps)
    if x.ndim == 1:
        return x / x.sum()
    return x / x.sum(axis=1, keepdims=True)


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1):
    train_parts, val_parts, test_parts = [], [], []
    for _, user_df in df.sort_values("timestamp").groupby("user_id", sort=False):
        n = len(user_df)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_parts.append(user_df.iloc[:n_train])
        val_parts.append(user_df.iloc[n_train : n_train + n_val])
        test_parts.append(user_df.iloc[n_train + n_val :])
    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)
    test = pd.concat(test_parts).reset_index(drop=True)
    return train, val, test


def load_movielens(data_dir: Path):
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_COLS
    items = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        encoding="latin-1",
        names=item_cols,
        usecols=list(range(len(item_cols))),
    )
    return ratings, items


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


def fit_stat_biases(train: pd.DataFrame, mu: float):
    user_mean = train.groupby("user_id")["rating"].mean()
    item_mean = train.groupby("item_id")["rating"].mean()
    user_bias = (user_mean - mu).to_dict()
    item_bias = (item_mean - mu).to_dict()
    return user_bias, item_bias


def add_stat_columns(df: pd.DataFrame, user_bias: Dict[int, float], item_bias: Dict[int, float], mu: float):
    out = df.copy()
    out["bu_stat"] = out["user_id"].map(user_bias).fillna(0.0).astype(float)
    out["bi_stat"] = out["item_id"].map(item_bias).fillna(0.0).astype(float)
    out["residual_stat"] = out["rating"].astype(float) - mu - out["bu_stat"] - out["bi_stat"]
    return out


def build_item_genres(items: pd.DataFrame, item_to_idx: Dict[int, int]):
    genre_cols = [g for g in GENRE_COLS if g != "unknown"]
    genre_to_idx = {g: i for i, g in enumerate(genre_cols)}
    item_genre_lists: Dict[int, List[int]] = {}
    item_genre_matrix = np.zeros((len(item_to_idx), len(genre_cols)), dtype=np.float64)
    for _, row in items.iterrows():
        item_id = int(row["item_id"])
        if item_id not in item_to_idx:
            continue
        genres = []
        for g in genre_cols:
            if int(row[g]) == 1:
                genres.append(genre_to_idx[g])
        if not genres:
            genres = []
        item_genre_lists[item_id] = genres
        if genres:
            item_genre_matrix[item_to_idx[item_id], genres] = 1.0 / len(genres)
    return genre_cols, item_genre_lists, item_genre_matrix


def robust_quantile_bins(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        lo, hi = float(values.min()), float(values.max())
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        edges = np.linspace(lo, hi, n_bins + 1)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def histogram(values: np.ndarray, centers: np.ndarray, edges: Optional[np.ndarray] = None, alpha: float = 1.0):
    if edges is None:
        half = np.diff(centers).mean() / 2.0 if len(centers) > 1 else 0.5
        edges = np.concatenate([[centers[0] - half], 0.5 * (centers[1:] + centers[:-1]), [centers[-1] + half]])
    counts, _ = np.histogram(values, bins=edges)
    return normalize_probs(counts.astype(np.float64) + alpha)


def fit_quantile_thresholds(values: pd.Series, n_bins: int) -> np.ndarray:
    raw = values.to_numpy(dtype=np.float64)
    thresholds = np.quantile(raw, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return np.unique(thresholds)


def apply_threshold_bins(values: pd.Series, thresholds: np.ndarray) -> pd.Series:
    raw = values.to_numpy(dtype=np.float64)
    return pd.Series(np.searchsorted(thresholds, raw, side="right"), index=values.index, dtype=int)


def simple_kmeans(x: np.ndarray, k: int, seed: int, n_iter: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(x) < k:
        return np.arange(len(x), dtype=np.int64)
    init_idx = rng.choice(len(x), size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(n_iter):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(axis=0)
            else:
                centers[j] = x[rng.integers(0, len(x))]
    return labels


def build_user_residual_features(
    train: pd.DataFrame,
    item_genres: Dict[int, List[int]],
    n_genres: int,
    beta: float = 10.0,
    min_count: float = 2.0,
):
    users = np.array(sorted(train["user_id"].unique()), dtype=np.int64)
    user_to_row = {u: i for i, u in enumerate(users)}
    sums = np.zeros((len(users), n_genres), dtype=np.float64)
    counts = np.zeros((len(users), n_genres), dtype=np.float64)
    for row in train.itertuples(index=False):
        genres = item_genres.get(int(row.item_id), [])
        if not genres:
            continue
        w = 1.0 / len(genres)
        idx = user_to_row[int(row.user_id)]
        for g in genres:
            sums[idx, g] += w * float(row.residual_stat)
            counts[idx, g] += w
    avg = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    shrink = counts / (counts + beta)
    features = shrink * avg
    features[counts < min_count] = 0.0
    return users, features, counts


def build_item_type_features(train: pd.DataFrame, n_items: int, item_genre_matrix: np.ndarray):
    item_stats = train.groupby("item_idx")["rating"].agg(["count", "mean"]).reindex(range(n_items)).fillna(0.0)
    count = np.log1p(item_stats["count"].to_numpy())[:, None]
    mean = item_stats["mean"].to_numpy()[:, None]
    x = np.hstack([item_genre_matrix, count, mean])
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return (x - x.mean(axis=0)) / std


def attach_group_column(
    splits: Iterable[pd.DataFrame],
    name: str,
    group_map: Dict[Tuple[int, int], int],
    key_cols: Tuple[str, str],
) -> None:
    col = f"{name}_group_idx"
    for df in splits:
        df[col] = [
            group_map.get((int(a), int(b)), 0)
            for a, b in zip(df[key_cols[0]].to_numpy(), df[key_cols[1]].to_numpy())
        ]


def build_prior_specs(
    cfg: ExperimentConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    items: pd.DataFrame,
    user_to_idx: Dict[int, int],
    item_to_idx: Dict[int, int],
) -> List[PriorSpec]:
    specs: List[PriorSpec] = []
    rng = np.random.default_rng(42)
    genre_cols, item_genres, item_genre_matrix = build_item_genres(items, item_to_idx)
    n_genres = len(genre_cols)

    rating_centers = np.arange(1, 6, dtype=np.float64)
    rating_probs = train["rating"].value_counts().sort_index().reindex(range(1, 6), fill_value=0).to_numpy()
    specs.append(PriorSpec("global_rating", "rating", rating_centers, rating_probs, "batch"))

    residual_edges, residual_centers = robust_quantile_bins(train["residual_stat"].to_numpy(), cfg.residual_bins)
    global_resid = histogram(train["residual_stat"].to_numpy(), residual_centers, residual_edges)
    specs.append(PriorSpec("global_residual", "residual", residual_centers, global_resid, "batch"))

    # Activity/popularity conditional residual prior.
    user_counts = train.groupby("user_id").size()
    item_counts = train.groupby("item_id").size()
    user_count_values = train["user_id"].map(user_counts).fillna(0)
    item_count_values = train["item_id"].map(item_counts).fillna(0)
    user_count_thresholds = fit_quantile_thresholds(user_count_values, 4)
    item_count_thresholds = fit_quantile_thresholds(item_count_values, 4)
    for df in (train, val, test):
        df["user_activity_bin"] = apply_threshold_bins(
            df["user_id"].map(user_counts).fillna(0),
            user_count_thresholds,
        )
        df["item_popularity_bin"] = apply_threshold_bins(
            df["item_id"].map(item_counts).fillna(0),
            item_count_thresholds,
        )

    activity_groups = []
    for u_bin in sorted(train["user_activity_bin"].unique()):
        for i_bin in sorted(train["item_popularity_bin"].unique()):
            activity_groups.append((int(u_bin), int(i_bin)))
    activity_map = {g: idx for idx, g in enumerate(activity_groups)}
    attach_group_column(
        (train, val, test),
        "cond_activity_popularity",
        activity_map,
        ("user_activity_bin", "item_popularity_bin"),
    )
    activity_probs = grouped_residual_probs(
        train,
        "cond_activity_popularity_group_idx",
        len(activity_groups),
        residual_centers,
        residual_edges,
        fallback=global_resid,
    )
    specs.append(
        PriorSpec(
            "cond_activity_popularity",
            "residual",
            residual_centers,
            activity_probs,
            "sample",
            ["user_activity_bin", "item_popularity_bin"],
            "P(E | user activity bin, item popularity bin)",
        )
    )

    # User type by genre residual lift, item side by genre.
    users, user_features, user_counts_by_genre = build_user_residual_features(
        train, item_genres, n_genres
    )
    valid = user_counts_by_genre >= 2.0
    scores = user_features.copy()
    scores[~valid] = -np.inf
    fallback = np.argmax(user_counts_by_genre, axis=1)
    genre_type = np.where(np.isfinite(scores).any(axis=1), np.argmax(scores, axis=1), fallback)
    user_genre_type = {int(u): int(t) for u, t in zip(users, genre_type)}

    for df in (train, val, test):
        df["user_genre_type"] = df["user_id"].map(user_genre_type).fillna(0).astype(int)
        df["primary_genre"] = [
            item_genres.get(int(item_id), [0])[0] if item_genres.get(int(item_id), []) else 0
            for item_id in df["item_id"]
        ]
    genre_groups = [(u, g) for u in range(n_genres) for g in range(n_genres)]
    genre_map = {g: idx for idx, g in enumerate(genre_groups)}
    attach_group_column((train, val, test), "cond_user_genre", genre_map, ("user_genre_type", "primary_genre"))
    genre_probs = grouped_residual_probs(
        train,
        "cond_user_genre_group_idx",
        len(genre_groups),
        residual_centers,
        residual_edges,
        fallback=global_resid,
    )
    specs.append(
        PriorSpec(
            "cond_user_genre",
            "residual",
            residual_centers,
            genre_probs,
            "sample",
            ["user_genre_type", "primary_genre"],
            "P(E | residual-lift user genre type, movie primary genre)",
        )
    )

    # Latent user clusters crossed with item clusters.
    user_x = user_features.copy()
    std = user_x.std(axis=0)
    std[std == 0] = 1.0
    user_x = (user_x - user_x.mean(axis=0)) / std
    user_labels = simple_kmeans(user_x, cfg.user_clusters, seed=42)
    user_cluster = {int(u): int(label) for u, label in zip(users, user_labels)}

    item_x = build_item_type_features(train, len(item_to_idx), item_genre_matrix)
    item_labels = simple_kmeans(item_x, cfg.item_clusters, seed=42)
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
    item_cluster = {int(idx_to_item[idx]): int(label) for idx, label in enumerate(item_labels)}

    for df in (train, val, test):
        df["user_cluster"] = df["user_id"].map(user_cluster).fillna(0).astype(int)
        df["item_cluster"] = df["item_id"].map(item_cluster).fillna(0).astype(int)
    cluster_groups = [
        (u, i)
        for u in range(cfg.user_clusters)
        for i in range(cfg.item_clusters)
    ]
    cluster_map = {g: idx for idx, g in enumerate(cluster_groups)}
    attach_group_column((train, val, test), "cond_user_itemtype", cluster_map, ("user_cluster", "item_cluster"))
    cluster_probs = grouped_residual_probs(
        train,
        "cond_user_itemtype_group_idx",
        len(cluster_groups),
        residual_centers,
        residual_edges,
        fallback=global_resid,
    )
    specs.append(
        PriorSpec(
            "cond_user_itemtype",
            "residual",
            residual_centers,
            cluster_probs,
            "sample",
            ["user_cluster", "item_cluster"],
            "P(E | residual user cluster, item-type cluster)",
        )
    )

    # Score-spread and dot-product priors are trained after a baseline exists.  The
    # runner inserts them per seed because their target depends on baseline factors.
    if cfg.include_oracle:
        oracle = histogram(test["residual_stat"].to_numpy(), residual_centers, residual_edges)
        specs.append(PriorSpec("oracle_test_residual", "residual", residual_centers, oracle, "batch"))

    if cfg.max_candidates is not None:
        specs = specs[: cfg.max_candidates]
    return specs


def grouped_residual_probs(
    df: pd.DataFrame,
    group_col: str,
    n_groups: int,
    centers: np.ndarray,
    edges: np.ndarray,
    fallback: np.ndarray,
    alpha: float = 1.0,
):
    probs = np.zeros((n_groups, len(centers)), dtype=np.float64)
    for g in range(n_groups):
        values = df.loc[df[group_col] == g, "residual_stat"].to_numpy()
        if len(values) == 0:
            probs[g] = fallback
        else:
            probs[g] = histogram(values, centers, edges, alpha=alpha)
    return normalize_probs(probs)


def soft_bins_and_grad(values: np.ndarray, centers: np.ndarray, tau: float):
    logits = -((values[:, None] - centers[None, :]) ** 2) / tau
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    q /= q.sum(axis=1, keepdims=True)
    dlogits = -2.0 * (values[:, None] - centers[None, :]) / tau
    mean_dlogits = (q * dlogits).sum(axis=1, keepdims=True)
    dq_dx = q * (dlogits - mean_dlogits)
    return q, dq_dx


def prior_loss_and_grad(
    values: np.ndarray,
    prior: PriorSpec,
    batch_df: pd.DataFrame,
    cfg: ExperimentConfig,
):
    targets = prior.target_for_rows(batch_df)
    q, dq_dx = soft_bins_and_grad(values, prior.centers, cfg.soft_tau)
    eps = 1e-10

    if prior.mode == "batch" and not prior.is_grouped:
        q_mean = normalize_probs(q.mean(axis=0))
        p = prior.probs
        kl = float(np.sum(q_mean * np.log((q_mean + eps) / (p + eps))))
        diff = q_mean - p
        mmd = float(np.sum(diff * diff))
        peak = float((prior.centers[np.argmax(q_mean)] - prior.centers[np.argmax(p)]) ** 2)
        dloss_dqmean = cfg.lambda_kl * (np.log((q_mean + eps) / (p + eps)) + 1.0)
        dloss_dqmean += cfg.lambda_mmd * 2.0 * diff
        if cfg.lambda_peak > 0:
            peak_idx = int(np.argmax(q_mean))
            dloss_dqmean[peak_idx] += cfg.lambda_peak * 2.0 * (
                prior.centers[peak_idx] - prior.centers[int(np.argmax(p))]
            )
        dloss_dq = dloss_dqmean[None, :] / len(values)
        grad = np.sum(dloss_dq * dq_dx, axis=1)
        loss = cfg.lambda_kl * kl + cfg.lambda_mmd * mmd + cfg.lambda_peak * peak
        stats = {"prior_kl": kl, "prior_mmd": mmd, "prior_peak": peak}
        return loss, grad, stats

    # Sample-conditional KL + MMD.
    p = targets
    kl_terms = np.sum(q * np.log((q + eps) / (p + eps)), axis=1)
    diff = q - p
    mmd_terms = np.sum(diff * diff, axis=1)
    peak_terms = (prior.centers[np.argmax(q, axis=1)] - prior.centers[np.argmax(p, axis=1)]) ** 2

    dloss_dq = cfg.lambda_kl * (np.log((q + eps) / (p + eps)) + 1.0)
    dloss_dq += cfg.lambda_mmd * 2.0 * diff
    if cfg.lambda_peak > 0:
        peak_idx = np.argmax(q, axis=1)
        p_peak = prior.centers[np.argmax(p, axis=1)]
        dloss_dq[np.arange(len(values)), peak_idx] += cfg.lambda_peak * 2.0 * (
            prior.centers[peak_idx] - p_peak
        )
    grad = np.sum(dloss_dq * dq_dx, axis=1) / len(values)
    loss = cfg.lambda_kl * float(kl_terms.mean())
    loss += cfg.lambda_mmd * float(mmd_terms.mean())
    loss += cfg.lambda_peak * float(peak_terms.mean())
    stats = {
        "prior_kl": float(kl_terms.mean()),
        "prior_mmd": float(mmd_terms.mean()),
        "prior_peak": float(peak_terms.mean()),
    }
    return loss, grad, stats


def train_model(
    model: MatrixFactorization,
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: ExperimentConfig,
    prior: Optional[PriorSpec] = None,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    history = []
    n = len(train)
    users = train["user_idx"].to_numpy(np.int64)
    items = train["item_idx"].to_numpy(np.int64)
    ratings = train["rating"].to_numpy(np.float64)
    bu_stat = train["bu_stat"].to_numpy(np.float64)
    bi_stat = train["bi_stat"].to_numpy(np.float64)

    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(n)
        epoch_mse = []
        epoch_prior = []
        epoch_kl = []
        epoch_mmd = []
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            u = users[idx]
            i = items[idx]
            r = ratings[idx]
            pred, dot = model.predict(u, i, return_dot=True)
            err = pred - r
            mse = float(np.mean(err * err))
            grad_pred = 2.0 * err / len(idx)
            grad_dot = np.zeros_like(grad_pred)

            prior_loss = 0.0
            stats = {"prior_kl": 0.0, "prior_mmd": 0.0}
            if prior is not None:
                batch_df = train.iloc[idx]
                if prior.value_kind == "rating":
                    values = pred
                    loss, grad_value, stats = prior_loss_and_grad(values, prior, batch_df, cfg)
                    grad_pred += grad_value
                elif prior.value_kind == "residual":
                    values = pred - model.mu - bu_stat[idx] - bi_stat[idx]
                    loss, grad_value, stats = prior_loss_and_grad(values, prior, batch_df, cfg)
                    grad_pred += grad_value
                elif prior.value_kind == "dot":
                    values = dot
                    loss, grad_value, stats = prior_loss_and_grad(values, prior, batch_df, cfg)
                    grad_dot += grad_value
                else:
                    raise ValueError(f"Unknown prior value kind: {prior.value_kind}")
                prior_loss = float(loss)

            model.update(u, i, grad_pred, grad_dot, cfg.lr, cfg.l2)
            epoch_mse.append(mse)
            epoch_prior.append(prior_loss)
            epoch_kl.append(stats.get("prior_kl", 0.0))
            epoch_mmd.append(stats.get("prior_mmd", 0.0))

        val_metrics = rating_metrics(model, val)
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(epoch_mse)),
            "train_prior_loss": float(np.mean(epoch_prior)),
            "train_prior_kl": float(np.mean(epoch_kl)),
            "train_prior_mmd": float(np.mean(epoch_mmd)),
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_clipped_rmse": val_metrics["clipped_rmse"],
        }
        history.append(row)
    return model, pd.DataFrame(history)


def rating_metrics(model: MatrixFactorization, df: pd.DataFrame):
    pred = model.predict(df["user_idx"].to_numpy(np.int64), df["item_idx"].to_numpy(np.int64))
    truth = df["rating"].to_numpy(np.float64)
    clipped = np.clip(pred, 1.0, 5.0)
    return {
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "clipped_rmse": float(np.sqrt(np.mean((clipped - truth) ** 2))),
        "clipped_mae": float(np.mean(np.abs(clipped - truth))),
    }


def distribution_metrics(values: np.ndarray, prior: PriorSpec, df: pd.DataFrame, cfg: ExperimentConfig):
    q, _ = soft_bins_and_grad(values, prior.centers, cfg.soft_tau)
    if prior.mode == "batch" and not prior.is_grouped:
        pred_dist = normalize_probs(q.mean(axis=0))
        target = prior.probs
    else:
        pred_dist = normalize_probs(q.mean(axis=0))
        target = normalize_probs(prior.target_for_rows(df).mean(axis=0))
    return compare_distributions(pred_dist, target, prior.centers)


def compare_distributions(p: np.ndarray, q: np.ndarray, centers: np.ndarray):
    eps = 1e-10
    p = normalize_probs(p)
    q = normalize_probs(q)
    m = 0.5 * (p + q)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    mean_p = float(np.sum(p * centers))
    mean_q = float(np.sum(q * centers))
    var_p = float(np.sum(p * (centers - mean_p) ** 2))
    var_q = float(np.sum(q * (centers - mean_q) ** 2))
    return {
        "kl": float(np.sum(p * np.log((p + eps) / (q + eps)))),
        "js": float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))),
        "wasserstein": float(np.mean(np.abs(cdf_p - cdf_q))),
        "mmd_l2": float(np.sum((p - q) ** 2)),
        "peak_bin_error": float(abs(centers[np.argmax(p)] - centers[np.argmax(q)])),
        "mean_error": float(abs(mean_p - mean_q)),
        "var_error": float(abs(var_p - var_q)),
    }


def ranking_metrics(model: MatrixFactorization, train: pd.DataFrame, test: pd.DataFrame, n_items: int, k: int):
    train_seen = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    test_pos = test[test["rating"] >= 4].groupby("user_idx")["item_idx"].apply(set).to_dict()
    if not test_pos:
        return {"recall_at_k": np.nan, "ndcg_at_k": np.nan, "map_at_k": np.nan, "coverage_at_k": np.nan, "novelty_at_k": np.nan}

    item_pop = train.groupby("item_idx").size().reindex(range(n_items)).fillna(0).to_numpy()
    item_pop_prob = (item_pop + 1.0) / (item_pop.sum() + n_items)
    recalls, ndcgs, maps, novelty = [], [], [], []
    recommended = set()
    all_items = np.arange(n_items, dtype=np.int64)
    for user, positives in test_pos.items():
        seen = train_seen.get(user, set())
        candidates = np.array([i for i in all_items if i not in seen], dtype=np.int64)
        if len(candidates) == 0:
            continue
        users = np.full(len(candidates), int(user), dtype=np.int64)
        scores = model.predict(users, candidates)
        top_local = np.argsort(scores)[-k:][::-1]
        top_items = candidates[top_local]
        recommended.update(top_items.tolist())
        hits = np.array([1.0 if int(item) in positives else 0.0 for item in top_items])
        recalls.append(float(hits.sum() / max(1, len(positives))))
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float(np.sum(hits * discounts))
        ideal = float(np.sum(discounts[: min(len(positives), k)]))
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        precision_at_hits = []
        hit_count = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                hit_count += 1.0
                precision_at_hits.append(hit_count / rank)
        maps.append(float(np.mean(precision_at_hits)) if precision_at_hits else 0.0)
        novelty.append(float(np.mean(-np.log2(item_pop_prob[top_items]))))
    return {
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "map_at_k": float(np.mean(maps)),
        "coverage_at_k": float(len(recommended) / n_items),
        "novelty_at_k": float(np.mean(novelty)),
    }


def subgroup_metrics(model: MatrixFactorization, train: pd.DataFrame, test: pd.DataFrame):
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
        "cold_user_or_long_tail": (df["cold_user"] | df["long_tail_item"]).to_numpy(bool),
    }
    rows = []
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        metrics = rating_metrics(model, df.loc[mask])
        rows.append({"subgroup": name, "n": int(mask.sum()), **metrics})
    return pd.DataFrame(rows)


def build_dot_prior_from_model(model: MatrixFactorization, train: pd.DataFrame, cfg: ExperimentConfig) -> PriorSpec:
    _, dots = model.predict(
        train["user_idx"].to_numpy(np.int64),
        train["item_idx"].to_numpy(np.int64),
        return_dot=True,
    )
    edges, centers = robust_quantile_bins(dots, cfg.dot_bins)
    probs = histogram(dots, centers, edges)
    return PriorSpec("global_dot", "dot", centers, probs, "batch", description="P(p_u dot q_i)")


def run_campaign(cfg: ExperimentConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ratings, items = load_movielens(cfg.data_dir)
    train, val, test = temporal_split(ratings)
    train, val, test, user_to_idx, item_to_idx = add_indices(train, val, test)
    mu = float(train["rating"].mean())
    stat_user_bias, stat_item_bias = fit_stat_biases(train, mu)
    train = add_stat_columns(train, stat_user_bias, stat_item_bias, mu)
    val = add_stat_columns(val, stat_user_bias, stat_item_bias, mu)
    test = add_stat_columns(test, stat_user_bias, stat_item_bias, mu)

    specs = build_prior_specs(cfg, train, val, test, items, user_to_idx, item_to_idx)
    metrics_rows = []
    history_rows = []
    subgroup_rows = []
    prior_rows = []

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(cfg.data_dir),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_users": len(user_to_idx),
        "n_items": len(item_to_idx),
        "config": vars(cfg) | {"data_dir": str(cfg.data_dir), "out_dir": str(cfg.out_dir)},
    }
    (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for seed in cfg.seeds:
        print(f"\n=== seed {seed} ===")
        initial = MatrixFactorization(len(user_to_idx), len(item_to_idx), cfg.factors, mu, seed)
        baseline = initial.copy()
        baseline, base_hist = train_model(baseline, train, val, cfg, prior=None, seed=seed)
        base_metrics = collect_all_metrics("baseline", "none", seed, baseline, train, val, test, None, cfg, len(item_to_idx))
        metrics_rows.append(base_metrics)
        for _, row in base_hist.iterrows():
            history_rows.append({"seed": seed, "candidate": "none", "variant": "baseline", **row.to_dict()})
        sg = subgroup_metrics(baseline, train, test)
        sg["seed"] = seed
        sg["candidate"] = "none"
        sg["variant"] = "baseline"
        subgroup_rows.extend(sg.to_dict("records"))

        seed_specs = list(specs)
        seed_specs.insert(2, build_dot_prior_from_model(baseline, train, cfg))
        if cfg.max_candidates is not None:
            seed_specs = seed_specs[: cfg.max_candidates]

        for spec in seed_specs:
            print(f"candidate={spec.name}")
            save_prior_spec(spec, cfg.out_dir / "priors")
            variants = [
                ("real_prior", spec),
                ("shuffled_prior", spec.shuffled(np.random.default_rng(seed + 1001))),
                ("uniform_prior", spec.uniform()),
                ("random_prior", spec.random(np.random.default_rng(seed + 2002))),
            ]
            for variant, prior in variants:
                model = initial.copy()
                model, hist = train_model(model, train, val, cfg, prior=prior, seed=seed)
                metrics_rows.append(
                    collect_all_metrics(variant, spec.name, seed, model, train, val, test, spec, cfg, len(item_to_idx))
                )
                for _, row in hist.iterrows():
                    history_rows.append({"seed": seed, "candidate": spec.name, "variant": variant, **row.to_dict()})
                sg = subgroup_metrics(model, train, test)
                sg["seed"] = seed
                sg["candidate"] = spec.name
                sg["variant"] = variant
                subgroup_rows.extend(sg.to_dict("records"))

            prior_rows.append(prior_summary_row(spec))

        pd.DataFrame(metrics_rows).to_csv(cfg.out_dir / "metrics.csv", index=False)
        pd.DataFrame(history_rows).to_csv(cfg.out_dir / "history.csv", index=False)
        pd.DataFrame(subgroup_rows).to_csv(cfg.out_dir / "subgroups.csv", index=False)
        pd.DataFrame(prior_rows).drop_duplicates("candidate").to_csv(cfg.out_dir / "prior_catalog.csv", index=False)
        summarize_results(cfg.out_dir)

    return cfg.out_dir


def collect_all_metrics(
    variant: str,
    candidate: str,
    seed: int,
    model: MatrixFactorization,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    prior: Optional[PriorSpec],
    cfg: ExperimentConfig,
    n_items: int,
):
    row = {"seed": seed, "candidate": candidate, "variant": variant}
    row.update({f"val_{k}": v for k, v in rating_metrics(model, val).items()})
    row.update({f"test_{k}": v for k, v in rating_metrics(model, test).items()})
    row.update(ranking_metrics(model, train, test, n_items, cfg.top_k))
    if prior is not None:
        pred, dot = model.predict(test["user_idx"].to_numpy(np.int64), test["item_idx"].to_numpy(np.int64), True)
        if prior.value_kind == "rating":
            values = pred
        elif prior.value_kind == "residual":
            values = pred - model.mu - test["bu_stat"].to_numpy() - test["bi_stat"].to_numpy()
        else:
            values = dot
        dist = distribution_metrics(values, prior, test, cfg)
        row.update({f"dist_{k}": v for k, v in dist.items()})
    return row


def prior_summary_row(spec: PriorSpec):
    return {
        "candidate": spec.name,
        "value_kind": spec.value_kind,
        "mode": spec.mode,
        "n_bins": len(spec.centers),
        "n_groups": int(spec.probs.shape[0]) if spec.is_grouped else 1,
        "description": spec.description,
    }


def save_prior_spec(spec: PriorSpec, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{spec.name}.npz",
        name=np.array(spec.name),
        value_kind=np.array(spec.value_kind),
        mode=np.array(spec.mode),
        centers=spec.centers,
        probs=spec.probs,
        description=np.array(spec.description),
    )


def summarize_results(out_dir: Path):
    metrics_path = out_dir / "metrics.csv"
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)
    base = df[df["variant"] == "baseline"][["seed", "test_rmse", "dist_kl"]].rename(
        columns={"test_rmse": "baseline_test_rmse", "dist_kl": "baseline_dist_kl"}
    )
    prior = df[df["variant"].isin(["real_prior", "shuffled_prior", "uniform_prior", "random_prior"])].copy()
    prior = prior.merge(base[["seed", "baseline_test_rmse"]], on="seed", how="left")
    prior["rmse_gain_vs_baseline"] = prior["baseline_test_rmse"] - prior["test_rmse"]
    summary = (
        prior.groupby(["candidate", "variant"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_rmse_gain=("rmse_gain_vs_baseline", "mean"),
            mean_dist_kl=("dist_kl", "mean"),
            mean_dist_js=("dist_js", "mean"),
            mean_dist_mmd=("dist_mmd_l2", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_ndcg_at_k=("ndcg_at_k", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    real = summary[summary["variant"] == "real_prior"][["candidate", "mean_rmse_gain", "mean_dist_kl"]]
    shuf = summary[summary["variant"] == "shuffled_prior"][["candidate", "mean_rmse_gain", "mean_dist_kl"]].rename(
        columns={"mean_rmse_gain": "shuffled_rmse_gain", "mean_dist_kl": "shuffled_dist_kl"}
    )
    if not real.empty:
        winners = real.merge(shuf, on="candidate", how="left")
        winners["beats_shuffled_rmse"] = winners["mean_rmse_gain"] > winners["shuffled_rmse_gain"].fillna(-np.inf)
        winners["beats_shuffled_dist"] = winners["mean_dist_kl"] < winners["shuffled_dist_kl"].fillna(np.inf)
        winners["meaningful_candidate"] = (
            winners["beats_shuffled_rmse"]
            & winners["beats_shuffled_dist"]
            & (winners["mean_rmse_gain"] >= -0.002)
        )
        winners.sort_values(["meaningful_candidate", "mean_dist_kl", "mean_rmse_gain"], ascending=[False, True, False]).to_csv(
            out_dir / "winner_screen.csv", index=False
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run prior-first QIML recommender experiments.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/ml-100k"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/qiml_prior_experiments"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
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
    parser.add_argument("--dot_bins", type=int, default=32)
    parser.add_argument("--user_clusters", type=int, default=12)
    parser.add_argument("--item_clusters", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--include_oracle", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One seed, few epochs, few candidates for smoke tests.")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.quick:
        seeds = [0]
        args.epochs = min(args.epochs, 2)
        args.factors = min(args.factors, 12)
        args.max_candidates = 3 if args.max_candidates is None else args.max_candidates

    return ExperimentConfig(
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
        dot_bins=args.dot_bins,
        user_clusters=args.user_clusters,
        item_clusters=args.item_clusters,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        include_oracle=args.include_oracle,
    )


def main():
    random.seed(0)
    np.random.seed(0)
    cfg = parse_args()
    out_dir = run_campaign(cfg)
    print(f"\nWrote experiment outputs to {out_dir}")
    print("Key files: metrics.csv, summary.csv, winner_screen.csv, prior_catalog.csv")


if __name__ == "__main__":
    main()
