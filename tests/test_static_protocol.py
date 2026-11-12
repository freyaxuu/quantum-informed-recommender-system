from pathlib import Path

import numpy as np
import pandas as pd

from qiml_prior_experiments import (
    ExperimentConfig,
    MatrixFactorization,
    rating_metrics,
    static_protocol_split,
    train_model,
)
from svd_prior_search import summarize_selection_metrics
from evaluate_qprior_recommender import (
    paired_seed_comparisons,
    positive_gain_preservation,
)
from train_winning_qprior import fit_qcbm


def make_config(epochs=6):
    return ExperimentConfig(
        data_dir=Path("data/ml-100k"),
        out_dir=Path("/tmp/static-protocol-test"),
        seeds=[0],
        epochs=epochs,
        factors=2,
        batch_size=4,
        lr=0.08,
        l2=1e-5,
        lambda_kl=0.0,
        lambda_mmd=0.0,
        lambda_peak=0.0,
        soft_tau=0.25,
        residual_bins=4,
        dot_bins=4,
        user_clusters=2,
        item_clusters=2,
        top_k=2,
        max_candidates=None,
        include_oracle=False,
    )


def test_static_protocol_split_is_disjoint_and_chronological():
    rows = []
    for user in (1, 2):
        for offset in range(10):
            rows.append(
                {
                    "interaction_id": f"{user}-{offset}",
                    "user_id": user,
                    "item_id": offset + 1,
                    "rating": 3,
                    "timestamp": user * 100 + offset,
                }
            )
    splits = static_protocol_split(pd.DataFrame(rows))
    frames = [splits.train, splits.checkpoint_val, splits.selection_val, splits.confirmation]
    observed = [set(frame["interaction_id"]) for frame in frames]
    assert set.union(*observed) == {row["interaction_id"] for row in rows}
    assert sum(len(values) for values in observed) == len(set.union(*observed))
    for user in (1, 2):
        boundaries = [frame.loc[frame["user_id"] == user, "timestamp"] for frame in frames]
        assert all(left.max() < right.min() for left, right in zip(boundaries, boundaries[1:]))


def test_train_model_restores_lowest_validation_rmse_checkpoint():
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 0, 1],
            "item_idx": [0, 1, 0, 1, 2, 2],
            "rating": [5.0, 1.0, 1.0, 5.0, 4.0, 2.0],
            "bu_stat": np.zeros(6),
            "bi_stat": np.zeros(6),
        }
    )
    val = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [1, 0],
            "rating": [1.0, 1.0],
            "bu_stat": np.zeros(2),
            "bi_stat": np.zeros(2),
        }
    )
    model = MatrixFactorization(2, 3, 2, mu=3.0, seed=0)
    restored, history = train_model(model, train, val, make_config(), seed=0)
    selected = history.loc[history["selected"]]
    assert len(selected) == 1
    assert selected.iloc[0]["val_rmse"] == history["val_rmse"].min()
    assert np.isclose(rating_metrics(restored, val)["rmse"], history["val_rmse"].min())


def test_validation_screen_requires_beating_every_null():
    rows = []
    values = {
        "baseline": (1.0, np.nan),
        "real_prior": (0.90, 0.10),
        "shuffled_prior": (0.95, 0.20),
        "uniform_prior": (0.96, 0.21),
        "random_prior": (0.89, 0.22),
    }
    for seed in (0, 1):
        for variant, (rmse, dist) in values.items():
            rows.append(
                {
                    "seed": seed,
                    "candidate": "candidate" if variant != "baseline" else "none",
                    "variant": variant,
                    "selected_epoch": 2,
                    "selection_rmse": rmse,
                    "selection_dist_kl": dist,
                }
            )
    _, screen = summarize_selection_metrics(pd.DataFrame(rows))
    assert screen["beats_shuffled_rmse"]
    assert screen["beats_uniform_rmse"]
    assert not screen["beats_random_rmse"]
    assert not screen["beats_all_nulls_rmse"]
    assert not screen["meaningful_candidate"]


def test_validation_screen_rejects_a_less_harmful_but_negative_gain():
    rows = []
    for seed in (0, 1):
        for variant, rmse, dist in (
            ("baseline", 1.0, np.nan),
            ("real_prior", 1.01, 0.10),
            ("shuffled_prior", 1.02, 0.20),
            ("uniform_prior", 1.03, 0.21),
            ("random_prior", 1.04, 0.22),
        ):
            rows.append(
                {
                    "seed": seed,
                    "candidate": "candidate" if variant != "baseline" else "none",
                    "variant": variant,
                    "selected_epoch": 2,
                    "selection_rmse": rmse,
                    "selection_dist_kl": dist,
                }
            )
    _, screen = summarize_selection_metrics(pd.DataFrame(rows))
    assert screen["beats_all_nulls_rmse"]
    assert screen["beats_all_nulls_dist"]
    assert screen["mean_val_rmse_gain"] < 0.0
    assert not screen["meaningful_candidate"]


def test_negative_empirical_gain_cannot_count_as_preservation():
    assert np.isnan(positive_gain_preservation(-0.02, -0.01))
    assert np.isnan(positive_gain_preservation(0.02, 0.0))
    assert np.isclose(positive_gain_preservation(0.009, 0.01), 0.9)


def test_paired_comparisons_use_positive_as_target_improvement():
    rows = []
    for seed in range(4):
        for variant, rmse, dist_kl in (
            ("baseline", 1.0, 0.5),
            ("empirical_prior", 0.9, 0.4),
            ("qcbm_prior", 0.8, 0.3),
            ("finite_shot_qcbm_prior", 0.85, 0.35),
            ("shuffled_prior", 0.95, 0.45),
            ("uniform_prior", 0.96, 0.46),
            ("random_prior", 0.97, 0.47),
        ):
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "test_rmse": rmse,
                    "empirical_dist_kl": dist_kl,
                }
            )
    comparisons = paired_seed_comparisons(pd.DataFrame(rows))
    row = comparisons.loc[
        (comparisons["target_variant"] == "qcbm_prior")
        & (comparisons["comparator_variant"] == "baseline")
    ].iloc[0]
    assert row["mean_rmse_improvement"] > 0.0
    assert row["mean_dist_kl_improvement"] > 0.0
    assert bool(row["rmse_ci_excludes_zero"])
    assert bool(row["dist_kl_ci_excludes_zero"])


def test_qcbm_restores_lowest_post_update_fitting_checkpoint():
    target = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float64)
    learned, _, history, selection = fit_qcbm(
        target=target,
        n_qubits=2,
        layers=1,
        epochs=5,
        lr=0.08,
        seed=7,
    )
    selected = [row for row in history if row["selected"]]
    assert len(selected) == 1
    assert selection["best_epoch"] == min(history, key=lambda row: row["loss"])["epoch"]
    eps = 1e-10
    restored_loss = float(
        np.sum(learned * np.log((learned + eps) / (target + eps)))
        + np.sum((learned - target) ** 2)
    )
    assert np.isclose(restored_loss, selection["best_loss"], atol=1e-10)
