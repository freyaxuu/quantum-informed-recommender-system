# Reproduce Table 1: static 16 × 16 × 16 QCBM prior

This is the corrected static-v1 experiment whose ten-seed mean confirmation
RMSE is **1.0220140337265988** for baseline MF and **1.0211049603466156**
for the exact QCBM prior. It is distinct from the older 12 × 16 and 24 × 24
experiments. Run commands from the repository root.

The original empirical prior, QCBM parameters/distributions, selection manifest,
validation search records, and per-seed confirmation results are included.
MovieLens data is excluded. No repeat of the hyperparameter search is required.

Verification on 2026-11-12 replayed all 70 MF fits (seven variants × ten seeds):
every saved numeric metric and selected epoch matched to absolute tolerance
1e-12, with zero RMSE difference. The original 216 circuit parameters were
also checked by independent NumPy state-vector reconstruction, and the 8,000-shot
distribution matched seed 42. This verification reused the saved QCBM;
quantum training was not rerun. The machine-readable audit is
`results/static_v1_corrected/table1_reproduction_verification.json`.

## Exact configuration

| MF setting | Value |
|---|---|
| Model | Explicit-feedback biased MF, `mu + bu + bi + P[u] @ Q[i]` |
| Latent factors | **20** |
| Optimizer | Custom **NumPy Adam**, beta1=0.9, beta2=0.999, epsilon=1e-8, float64 |
| Learning rate | **0.01** |
| L2 | **0.0001**, coupled gradient penalty on factors and biases |
| Batch size | **1,024** |
| Epoch budget | **20**; restore each variant's lowest checkpoint-validation RMSE epoch |
| MF initialization and batch-shuffle seeds | **0,1,2,3,4,5,6,7,8,9** |
| Initialization | Factors sampled from Normal(0, 0.05²), zero learned biases |
| Prior loss | `0.03 * KL(q || p) + 0.10 * sum((q-p)**2)`, mean over batch |
| Soft bins | `softmax(-(predicted_residual - centers)**2 / 0.15)` |
| Peak penalty | 0 |
| Predictions | Unclipped for RMSE |

The L2 implementation adds `l2 * factor` and `l2 * bias` for each batch
occurrence before accumulation by ID. It is not AdamW or a directly
interchangeable PyTorch `weight_decay` setting. Use the supplied training loop
for numerical reproduction. Each variant starts from the same MF initialization
and uses the same batch ordering for its seed. MF factors are trained from
these seeds; the original evaluator did not save downstream MF weight arrays.

| QCBM setting | Value |
|---|---|
| Qubits / layers | **12 / 6** (216 parameters) |
| Simulator | PennyLane `default.qubit`, analytic probabilities, Torch interface |
| Circuit per layer | RX, RY, RZ on every wire, then CNOT 0→1→…→11 and 11→0 |
| Initialization | `torch.manual_seed(42)`; `0.01 * torch.randn(6,12,3,dtype=float64)` |
| Optimizer | Torch Adam, lr **0.02**, default betas/epsilon, weight_decay=0 |
| Epoch budget / selected epoch | **300 / 300** |
| Checkpoint criterion | Minimum **post-update** `KL(pred || target) + sum((pred-target)**2)` |
| QCBM seed | **42** |
| Finite-shot distribution | **8,000** multinomial shots, NumPy `default_rng(42)` |
| Exact checkpoint fitting KL / loss | 0.3137988806060846 / 0.313963249818862 |

The downstream MF weights 0.03 and 0.10 are separate from the QCBM fitting
objective, which uses unit coefficients. The reference commands use CPU
float64. GPU migration or optimizer substitutions may change numerical results.

## Data and split

Use Python 3.12 and the dependencies in `requirements.txt`. Replaying the
saved prior in MF needs only NumPy and pandas; retraining the QCBM also needs
Torch and PennyLane. The original run did not save a full runtime lock or
hardware fingerprint, so the requirements file alone is not proof of its
historical environment. The included checkpoints permit downstream reproduction
without dependency-sensitive quantum retraining.

The verified MF replay used Python 3.12.14, NumPy 2.3.5 and pandas 2.2.3.
For that downstream-only replay, install `numpy==2.3.5 pandas==2.2.3` in a
Python 3.12 environment. Torch/PennyLane are unnecessary when using the saved
QCBM probabilities. The separate QCBM retraining commands need their dependencies
from `requirements.txt`; its historical version pins were not independently
validated by this replay.

Put the standard MovieLens 100K `u.data` and `u.item` files in `data/ml-100k/`.
`static_protocol_split()` sorts by timestamp, then divides each user's sequence
into 70% train, 10% checkpoint validation, 10% selection validation, and remainder
confirmation, with integer rounding per user. Retain the supplied sorting code,
including its treatment of equal timestamps. User/item indices are learned
from train only; later rows with unknown IDs are excluded.

| Partition | Evaluable rows | Use |
|---|---:|---|
| Train | 69,572 | Fit MF, statistical biases, clusters and prior |
| Checkpoint validation | 9,557 | Select each MF run's epoch |
| Selection validation | 9,550 | Original hyperparameter/prior selection |
| Confirmation | 11,160 | Frozen-configuration Table 1 evaluation |

There are 943 users and 1,575 train-known items. The selection manifest records
SHA-256 fingerprints of the exact rows in each partition. The evaluator checks
these fingerprints and the rebuilt empirical prior before training.

## Which code constructs the conditional prior?

`svd_prior_search.py` supplies `load_context()`, `compute_user_clusters()`,
`compute_item_clusters()` and **`build_user_itemtype_prior()`**. Their helper
implementations are in `qiml_prior_experiments.py`.

1. Calculate train-only `mu`, `bu_stat = mean_user_rating - mu` and
   `bi_stat = mean_item_rating - mu`. The target residual is
   `E = rating - mu - bu_stat - bi_stat`; these statistical biases are distinct
   from the trainable MF biases.
2. Form 18-dimensional user genre residual features, dividing a movie's
   contribution equally across its genres. Shrink each mean by
   `count/(count+10)`; set features with weighted count <2 to zero. Standardize
   columns and cluster into 16 groups with the supplied NumPy k-means, seed 42,
   at most 50 iterations.
3. Form item features from the 18 normalized genre indicators, log1p training
   count and training mean rating. Standardize and cluster into 16 groups
   using the same k-means implementation and seed 42.
4. Make 16 global quantile bins from training residuals. Centers are adjacent
   edge midpoints; widen the extreme edges by 1e-6. Each user/item group gets
   its train-only residual histogram with add-one smoothing. Empty groups use
   the add-one-smoothed global training histogram.
5. Store the conditional table as **256 rows × 16 residual bins**. Row index
   is `user_cluster * 16 + item_cluster`. Flatten in NumPy C order and normalize
   globally to obtain the 4,096-state QCBM target. Thus each group has equal
   total target mass, rather than empirical group-occupancy weighting.
6. After QCBM fitting, reshape to 256 × 16 and normalize each row before
   applying the conditional prior to MF training.

Verify and rebuild this exact prior without running model selection:

```bash
python3 rebuild_static_table1_prior.py --verify_only
python3 rebuild_static_table1_prior.py \
  --out_dir results/static_v1_corrected/rebuilt_table1_prior
```

The second command exports a matching `.npz` and verification JSON. Both commands
compare against the original frozen prior and split hashes. The original
selected prior remains the input to the commands below.

## Reproduce Table 1 using the saved QCBM

```bash
python3 evaluate_qprior_recommender.py \
  --selection_manifest results/static_v1_corrected/svd_prior_search_common_reference/best_config/selected/selection_manifest.json \
  --data_dir data/ml-100k \
  --qcbm_prior models/qprior_winning/static_v1_corrected/cond_user_itemtype/qcbm_prior.npy \
  --finite_shot_prior models/qprior_winning/static_v1_corrected/cond_user_itemtype/qcbm_prior_finite_shots.npy \
  --out_dir results/static_v1_corrected/table1_reproduction \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --batch_size 1024 --residual_bins 16 --lambda_peak 0 --top_k 10
```

The evaluator loads factors=20, lr=0.01, l2=1e-4, epochs=20, lambda_kl=0.03,
lambda_mmd=0.10, soft_tau=0.15 and 16×16 grouping **from the selection
manifest**, overriding those CLI settings. Use a new output directory for every
run; the original confirmation directory already contains the published results.

| Variant | Expected mean confirmation RMSE |
|---|---:|
| Baseline | 1.0220140337265988 |
| Empirical prior | 1.0212351147193360 |
| Exact QCBM | 1.0211049603466156 |
| 8,000-shot QCBM | 1.0206761535105588 |
| Shuffled | 1.0213721344781361 |
| Uniform | 1.0213544750357790 |
| Random | 1.0213004616083340 |

Shuffled controls permute conditional rows with seed `MF_seed + 1001`.
Random controls draw `Uniform(0,1) + 1e-3` per cell and row-normalize, using
seed `MF_seed + 2002`. Uniform controls use 1/16 for each bin.

## Retrain the six-layer QCBM or sweep depths on HPC

The exact command for the original QCBM settings is:

```bash
python3 train_winning_qprior.py \
  --prior results/static_v1_corrected/svd_prior_search_common_reference/best_config/selected/priors/cond_user_itemtype.npz \
  --out_dir models/qprior_winning/static_table1_retrained/layers_6 \
  --n_qubits 12 --layers 6 --epochs 300 --lr 0.02 --seed 42 --shots 8000
```

Use the resulting `qcbm_prior.npy` and `qcbm_prior_finite_shots.npy` paths
in the evaluator command above. Keep `target_prior.npy` and `qcbm_report.json`
beside them, because the evaluator verifies their provenance and checkpoint loss.

Example exploratory depth sweep, with all non-depth settings held fixed:

```bash
for depth in 2 4 6 8 10 12; do
  model_dir="models/qprior_winning/static_table1_depths/layers_${depth}"
  python3 train_winning_qprior.py \
    --prior results/static_v1_corrected/svd_prior_search_common_reference/best_config/selected/priors/cond_user_itemtype.npz \
    --out_dir "$model_dir" \
    --n_qubits 12 --layers "$depth" --epochs 300 --lr 0.02 --seed 42 --shots 8000
  python3 evaluate_qprior_recommender.py \
    --selection_manifest results/static_v1_corrected/svd_prior_search_common_reference/best_config/selected/selection_manifest.json \
    --qcbm_prior "$model_dir/qcbm_prior.npy" \
    --finite_shot_prior "$model_dir/qcbm_prior_finite_shots.npy" \
    --out_dir "results/static_v1_corrected/table1_depths/layers_${depth}" \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --batch_size 1024 --residual_bins 16 --lambda_peak 0 --top_k 10
done
```

A depth sweep on these already inspected confirmation rows is exploratory.
Selecting a new depth for a confirmatory claim needs validation followed by a
fresh locked holdout. The current improvement is small and simulator-based;
it does not establish quantum advantage.

## Original evidence and source map

- `qiml_prior_experiments.py`: MF, NumPy Adam, train loop, checkpointing,
  chronological partitions, residual features, histogram/soft-bin losses.
- `svd_prior_search.py`: prior construction and original validation-only search.
- `train_winning_qprior.py`: circuit architecture, fit, checkpoint and shot sampling.
- `evaluate_qprior_recommender.py`: frozen-prior downstream evaluation and null controls.
- `results/static_v1_corrected/svd_baseline_grid/`: original baseline search.
- `results/static_v1_corrected/svd_prior_search_common_reference/`: search log,
  candidate validation metrics and frozen `best_config/selected/` prior/manifest.
- `models/qprior_winning/static_v1_corrected/cond_user_itemtype/`: original
  `qcbm_state.npz`, target, exact/shot distributions and full fitting history.
- `results/static_v1_corrected/qiml_qprior_confirmation/`: original `metrics.csv`,
  `history.csv` (selected epochs), `summary.csv`, null comparisons and manifest.
- `tests/test_static_protocol.py`: split/checkpoint/null-control protocol checks.
