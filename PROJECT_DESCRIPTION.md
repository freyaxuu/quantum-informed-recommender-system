# Project Description: QCBM Priors for Recommender Systems

## Goal

This project studies whether a Quantum Circuit Born Machine (QCBM) can learn a compact, useful prior distribution for recommender systems, following the prior-learning pattern in *Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos* by Wang et al.

The paper's central idea is not to encode raw training data directly into a quantum circuit. Instead, the quantum circuit learns a compact statistical distribution that captures stable structure in the data. That learned distribution is then frozen and used as a prior during downstream model training.

This project adapts that idea to recommendation:

1. Discover a meaningful empirical recommender prior using classical experiments.
2. Train a QCBM to approximate that empirical prior.
3. Freeze the QCBM-learned prior.
4. Re-test the recommender with the empirical prior, QCBM prior, finite-shot QCBM prior, and null controls.
5. Make a QIML-style claim only if the QCBM prior preserves the empirical prior's benefit and beats shuffled/random/uniform controls.

## Research Question

What is the recommender-system analogue of the invariant measure used in the Wang et al. paper?

For MovieLens recommendation, the working hypothesis has been that useful priors may live in distributions such as:

- residual ratings conditioned on user and item structure;
- temporal next-residual distributions conditioned on recent user state;
- user taste-state transition distributions;
- residual autocorrelation statistics over user rating sequences.

The current evidence suggests that simple residual priors can improve RMSE slightly, but null controls remain very competitive. Therefore, the project is currently focused on diagnosing which prior contains specific recommender signal rather than generic regularization.

## Dataset

The first experimental campaign uses MovieLens 100K:

```text
data/ml-100k/
```

The code expects the standard `u.data` and item metadata files under that directory.

## Current Pipeline

### Stage 1: Classical Prior Discovery

Implemented in:

```text
qiml_prior_experiments.py
```

This stage tests candidate priors using classical Matrix Factorization before involving quantum circuits.

Main candidates tested:

- global rating prior;
- global residual prior;
- global dot-product prior;
- conditional residual prior by user activity and item popularity;
- conditional residual prior by user genre type;
- conditional residual prior by user cluster and item-type cluster.

The best classical candidate so far is:

```text
cond_user_itemtype = P(E | residual user cluster, item-type cluster)
```

where `E = rating - global_mean - user_bias - item_bias`.

### Stage 1.5: SVD Tuning and Grouping/Time Search

Implemented in:

```text
svd_prior_search.py
```

Before extending Stage 1 to a time axis, this stage first tabulates the
plain (no-prior) SVD/MF model across its own hyperparameters
(`results/svd_baseline_grid/`), since the earlier Stage 1 default
(`factors=50, lr=0.003, l2=1e-5`) turned out to be under-regularized.
A better-tuned baseline (`factors=20, lr=0.001, l2=1e-4`) drops test RMSE
from `1.146` to `1.002` with no prior at all, which shrinks the room for any
residual prior to contribute.

It then runs a staged search (`results/svd_prior_search/`) over user-cluster
count, item-cluster count, a new weekday/weekend and morning/afternoon/
evening time-bucket axis (timestamps were previously only used for
chronological ordering, never as a grouping feature), and the prior-loss
weights (`lambda_kl`, `lambda_mmd`, `soft_tau`). With the tuned SVD baseline,
`cond_user_itemtype` at finer granularity (`user_clusters=24,
item_clusters=24`, higher `lambda_kl=0.03`/`lambda_mmd=0.1`, lower
`soft_tau=0.15`) is the first configuration found to beat all three null
controls -- shuffled, uniform, and random -- simultaneously on both RMSE
gain and distribution KL, across a full 10-seed protocol. The margin remains
small (about `0.1%` of baseline RMSE), so this is still a weak positive
result, but a more robustly established one than the original 12x16 config.

### Stage 2: QCBM Prior Training

Implemented in:

```text
train_winning_qprior.py
evaluate_qprior_recommender.py
```

The selected empirical prior has shape:

```text
192 x 16 = 3072 probability states
```

The trained QCBM uses:

```text
12 qubits
6 layers
216 trainable parameters
```

The compressed QCBM parameter representation is much smaller than the empirical table:

```text
empirical prior table: 24,576 bytes
QCBM parameters:       1,728 bytes
```

This original run did not clear the project's `stage2_success` bar: the
QCBM's RMSE gain fell just short of the shuffled control.

Stage 2 was re-run against the Stage 1.5 winner
(`cond_user_itemtype`, `user_clusters=24, item_clusters=24`, tuned SVD) with
a larger circuit (14 qubits, 9,216-state table, 36.6x compression). At that
config, the exact-simulated QCBM prior clears `stage2_success` for the first
time: it preserves 96.2% of the empirical prior's RMSE gain and beats
shuffled, uniform, and random simultaneously on both RMSE gain and
distribution KL. The finite-shot (8,000-shot) version does not clear the
bar, because the larger table dilutes the same shot budget more than the
original 3,072-state prior did -- so this remains a simulator-only result,
not a hardware-ready one, and the RMSE margin is still small in absolute
terms (~0.09% of baseline RMSE). See `HANDOFF.md`'s "Stage 2 Re-Test on the
Tuned SVD + 24x24 Config" section for the full comparison table.

### Stage 3: Temporal Prior Experiments

Implemented in:

```text
temporal_qiml_experiments.py
```

This stage treats each user's ratings as a timestamp-ordered sequence. It tests whether temporal priors are more meaningful than static residual priors.

Main temporal priors:

- `temporal_next_residual`: `P(E_t | S_{t-1}, item_type_t)`
- `temporal_delta_residual`: `P(delta E_t | E_{t-1 bin}, S_{t-1}, item_type_t)`
- `temporal_gap_residual`: `P(E_t | gap_bucket, S_{t-1}, item_type_t)`
- `temporal_global_residual`: global temporal residual control

The runner also saves train-only diagnostic targets:

```text
diagnostic_priors/state_occupancy.npz
diagnostic_priors/state_transition.npz
diagnostic_priors/residual_autocorr.npz
```

These are closer to the paper's invariant-measure idea, but the transition and occupancy priors are currently diagnostics rather than direct training losses.

## Current Conclusion

The project has not yet produced a strong paper-style QIML result.

The strongest current result is a weak positive one:

- the empirical static prior `P(E | user_cluster, item_cluster)` improves RMSE;
- with a properly tuned SVD baseline and finer grouping (`user_clusters=24,
  item_clusters=24`), this prior now beats shuffled, uniform, and random
  controls simultaneously (Stage 1.5), which is a more robust screening
  result than the original 12x16 config;
- the QCBM re-test was re-run against this tuned config (Stage 2, above) and,
  for the first time, the exact-simulated QCBM prior also beats shuffled,
  uniform, and random simultaneously while preserving 96.2% of the
  empirical prior's RMSE gain;
- however, the RMSE margin is still small (about `0.1%` of baseline RMSE for
  the empirical prior, `0.09%` for the QCBM prior), and the finite-shot
  (8,000-shot) QCBM does not clear the same bar -- so this is a stronger and
  more robustly established weak positive, not evidence of a hardware-ready
  or "strong" QIML result.

The main research bottleneck is not QCBM training. It is prior discovery.

The next major step should be to find a prior whose benefit is clearly larger than seed noise and clearly stronger than null controls. The most promising direction is a sequence model that directly predicts or rolls out taste states, then uses `P(S_t | S_{t-1})` or state occupancy as the prior.

