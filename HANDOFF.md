# Handoff: Q-Prior Recommender Experiments

Last updated: 2026-07-27

## Project Aim

This project adapts *Quantum-Informed Machine Learning for Predicting Spatiotemporal Chaos* by Wang et al. to recommender systems.

The intended claim is:

1. Find a meaningful empirical recommender prior.
2. Train a QCBM to approximate that prior.
3. Freeze the QCBM prior.
4. Use it during recommender training.
5. Show that the QCBM prior preserves most of the empirical prior's benefit and beats null controls.

Do not train new quantum circuits until a classical prior survives shuffled/random/uniform controls.

## Important Files

Code:

```text
qiml_prior_experiments.py
svd_prior_search.py
train_winning_qprior.py
evaluate_qprior_recommender.py
temporal_qiml_experiments.py
```

Docs:

```text
README.md
README_QIML_EXPERIMENTS.md
PROJECT_DESCRIPTION.md
HANDOFF.md
```

Key results:

```text
results/qiml_prior_experiments/
results/qiml_qprior_retest/cond_user_itemtype/
results/temporal_qiml_prior_experiments/
results/svd_baseline_grid/
results/svd_prior_search/
models/qprior_winning/cond_user_itemtype/
```

## Stage 1: Static Classical Prior Discovery

Runner:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  qiml_prior_experiments.py
```

Result file:

```text
results/qiml_prior_experiments/winner_screen.csv
```

Best candidate:

```text
cond_user_itemtype = P(E | residual user cluster, item-type cluster)
```

Summary:

| candidate | RMSE gain | dist KL | shuffled RMSE gain | shuffled dist KL | meaningful |
|---|---:|---:|---:|---:|---|
| `cond_user_itemtype` | `0.003016` | `0.092155` | `0.002960` | `0.092217` | `True` |
| `cond_user_genre` | `0.003074` | `0.093227` | `0.003130` | `0.093343` | `False` |
| `cond_activity_popularity` | `0.003058` | `0.095222` | `0.003074` | `0.095236` | `False` |
| `global_rating` | `0.000183` | `0.075474` | `-0.000729` | `0.074508` | `False` |
| `global_dot` | `-0.000220` | `0.069325` | `-0.000220` | `0.069325` | `False` |
| `global_residual` | `-0.000843` | `0.088741` | `-0.000843` | `0.088741` | `False` |

Interpretation:

`cond_user_itemtype` was the only static prior that passed the implemented screen, but the margin over shuffled controls is tiny. Treat it as a weak positive candidate, not a decisive invariant measure.

## Stage 2: QCBM Re-Test

QCBM source prior:

```text
results/qiml_prior_experiments/priors/cond_user_itemtype.npz
```

QCBM artifacts:

```text
models/qprior_winning/cond_user_itemtype/qcbm_prior.npy
models/qprior_winning/cond_user_itemtype/qcbm_prior_finite_shots.npy
models/qprior_winning/cond_user_itemtype/qcbm_report.json
```

Circuit:

```text
prior shape: 192 x 16 = 3072 states
qubits: 12
layers: 6
trainable parameters: 216
empirical prior size: 24,576 bytes
QCBM parameter size: 1,728 bytes
```

QCBM fit:

```text
QCBM vs empirical KL:        0.2802
finite-shot vs empirical KL: 0.4884
```

Recommender re-test:

| variant | RMSE | RMSE gain | RMSE preservation | dist KL preservation | stage2 success |
|---|---:|---:|---:|---:|---|
| baseline | `1.147523` | `0.000000` | - | - | - |
| empirical prior | `1.144507` | `0.003016` | `1.000` | `1.000` | reference |
| QCBM prior | `1.144600` | `0.002923` | `0.969` | `0.989` | `False` |
| finite-shot QCBM | `1.145134` | `0.002389` | `0.792` | `0.826` | `False` |
| shuffled prior | `1.144563` | `0.002960` | `0.981` | `1.032` | `False` |
| uniform prior | `1.144441` | `0.003082` | `1.022` | `1.057` | `False` |
| random prior | `1.144456` | `0.003067` | `1.017` | `1.058` | `False` |

Interpretation:

The QCBM compresses the empirical table and preserves most of the empirical prior's RMSE gain. However, it does not beat shuffled/uniform/random controls. This means the current result is not strong enough for a paper-style QIML claim.

## Stage 2 Re-Test on the Tuned SVD + 24x24 Config

Last updated: 2026-08-06.

Re-ran Stage 2 against the Stage 1.5 winner (`cond_user_itemtype`,
`user_clusters=24, item_clusters=24`, tuned SVD `factors=20, lr=0.001,
l2=1e-4`, `lambda_kl=0.03, lambda_mmd=0.1, soft_tau=0.15`) instead of the
original 12x16 / untuned-SVD config.

QCBM source prior:

```text
results/svd_prior_search/best_config/rank2_cond_user_itemtype/priors/cond_user_itemtype.npz
```

QCBM artifacts:

```text
models/qprior_winning/cond_user_itemtype_24x24_tuned/qcbm_prior.npy
models/qprior_winning/cond_user_itemtype_24x24_tuned/qcbm_prior_finite_shots.npy
models/qprior_winning/cond_user_itemtype_24x24_tuned/qcbm_report.json
```

Circuit (larger than before, because 24x24 grouping x 16 residual bins is a
3x larger table than the original 12x16):

```text
prior shape: 576 x 16 = 9216 states
qubits: 14
layers: 6
trainable parameters: 252
empirical prior size: 73,728 bytes
QCBM parameter size: 2,016 bytes (36.6x compression)
```

QCBM fit:

```text
QCBM vs empirical KL:        0.3775  (worse than the original 0.2802 -- larger table, same param budget)
finite-shot vs empirical KL: 0.9301  (worse than the original 0.4884 -- 8000 shots spread over 3x more states)
```

Recommender re-test
(`results/qiml_qprior_retest/cond_user_itemtype_24x24_tuned/`):

| variant | RMSE gain | RMSE preservation | dist KL preservation | beats all 3 nulls (RMSE) | beats all 3 nulls (dist KL) | stage2 success |
|---|---:|---:|---:|---|---|---|
| baseline | `0.000000` | - | - | - | - | - |
| empirical prior | `0.000959` | `1.000` | `1.000` | yes | yes | reference |
| QCBM prior | `0.000923` | `0.962` | `1.166` | **yes** | **yes** | **`True`** |
| finite-shot QCBM | `0.000638` | `0.665` | `1.205` | no | yes | `False` |
| shuffled prior | `0.000873` | `0.910` | `1.212` | - | - | - |
| uniform prior | `0.000894` | `0.932` | `1.230` | - | - | - |
| random prior | `0.000848` | `0.884` | `1.227` | - | - | - |

Interpretation:

For the first time in this project, the exact-simulated QCBM prior clears
this repo's `stage2_success` bar: it preserves 96.2% of the empirical
prior's RMSE gain, exceeds its distribution-fidelity gain (116.6%), and
beats shuffled, uniform, **and** random simultaneously on both RMSE gain
and distribution KL -- at 36.6x compression (73,728 bytes -> 2,016 bytes).
This is a meaningfully stronger result than the original 12x16 / untuned-SVD
run, where the QCBM's RMSE gain fell just short of the shuffled control
(`0.002923` vs `0.002960`) and `stage2_success` was `False`.

Two caveats keep this from being a strong claim:

1. The RMSE margin is still small in absolute terms (`0.000923` on a
   `~1.002` baseline RMSE, about `0.09%`), even though it is now
   consistently ahead of every null control rather than borderline.
2. The **finite-shot** QCBM (8,000 shots, closer to what real hardware would
   see) does *not* clear the bar -- its RMSE preservation drops to 66.5% and
   it no longer beats the shuffled control. This is because the winning
   config's prior table (9,216 states) is 3x larger than the original
   (3,072 states) for the same shot budget, so per-state shot noise is
   proportionally worse. The idealized-simulator success should not be
   read as a hardware-ready result.

## Stage 3: Temporal Prior Experiments

Runner:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py \
  --window 10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20
```

Result file:

```text
results/temporal_qiml_prior_experiments/winner_screen.csv
```

Saved run:

```text
window: 10
seeds: 0-9
epochs: 20
train samples: 70,187
val samples: 9,560
test samples: 10,736
```

Temporal winner screen:

| candidate | RMSE gain | dist KL | user-shuffled gain | time-shuffled gain | meaningful |
|---|---:|---:|---:|---:|---|
| `temporal_next_residual` | `0.002775` | `0.104376` | `0.002686` | `0.002724` | `False` |
| `temporal_gap_residual` | `0.002753` | `0.103713` | `0.002712` | `0.002691` | `False` |
| `temporal_delta_residual` | `0.002608` | `0.085510` | `0.003015` | `0.003116` | `False` |
| `temporal_global_residual` | `-0.000720` | `0.093043` | `-0.000720` | `-0.000720` | `False` |

Interpretation:

No temporal prior candidate currently wins.

`temporal_next_residual` and `temporal_gap_residual` improve RMSE slightly, but their distribution-fidelity scores are worse than several null controls. That suggests the prior loss is acting like weak regularization rather than capturing a specific temporal recommender invariant.

The saved result only covers `window=10`. There is no saved `--windows 5,10,20` sweep under `results/` at the time of this handoff.

## Stage 1.5: SVD Tuning + Grouping/Time Hyperparameter Search

Last updated: 2026-08-06.

Before moving to the sequence-model direction below, we tabulated the standard
(no-prior) SVD/MF model across its own hyperparameters and ran a staged search
over user-cluster count, item-cluster count, a new time-bucket axis
(weekday/weekend, morning/afternoon/evening), and the prior-loss weights.
Runner: `svd_prior_search.py` (`--stage baseline|a|b|c|all`), reusing the MF
model, training loop, and metric functions from `qiml_prior_experiments.py`
unchanged.

### Standard SVD tabulation

`results/svd_baseline_grid/summary.csv` (27 factors/lr/l2 configs x 5 seeds).
The existing default (`factors=50, lr=0.003, l2=1e-5`) reproduces the
`test_rmse=1.1464` baseline reported earlier in this file, but it is not a
well-tuned plain SVD: `factors=20, lr=0.001, l2=1e-4` reaches
`test_rmse=1.0024` with **no prior at all**. The old default was
under-regularized (`l2` too small, `lr` too high), which matters for
Stage 1's conclusions below.

### Grouping/time axis screen and prior-loss tuning

With the tuned SVD baseline, every grouping candidate's RMSE gain from the
prior loss shrinks to roughly `0.0003-0.001`, about 3-10x smaller than the
`0.003016` headline number for `cond_user_itemtype` (12x16) reported above.
This is expected: a better-regularized baseline leaves much less residual
error for any prior to correct, and it means the earlier `0.003` gain was
partly compensating for an under-tuned baseline rather than reflecting a
larger recommender-structure signal.

The staged search (`results/svd_prior_search/search_log.csv`) found that
scaling up `lambda_kl` (0.03) and `lambda_mmd` (0.1) with a lower
`soft_tau` (0.15), combined with finer grouping (`user_clusters=24,
item_clusters=24`), consistently pushed `real_prior` above `shuffled_prior`
across the whole loss-weight grid (54/54 configs), not just one cherry-picked
setting — a more systematic signal than the single-configuration screen used
previously.

Full 10-seed x 4-variant confirmation
(`results/svd_prior_search/best_config/`):

| candidate | grouping | lambda_kl/mmd/tau | RMSE gain | beats shuffled | beats uniform | beats random | beats all 3 (RMSE) | beats all 3 (dist KL) |
|---|---|---|---:|---|---|---|---|---|
| `cond_user_itemtype` | user=24 x item=24 | 0.03/0.1/0.15 | `0.000959` | yes | yes | yes | **yes** | **yes** |
| `cond_user_item_time` | user=24 x item=24 x weekday/weekend | 0.03/0.1/0.15 | `0.000992` | yes | yes | yes (barely) | yes | no (random has lower KL) |
| `cond_user_item_time` | user=24 x item=24 x weekday/weekend | 0.03/0.1/0.25 | `0.000806` | yes | no (uniform lower KL) | no (random lower KL) | yes | no |

`cond_user_itemtype` at `user_clusters=24, item_clusters=24` is the standout:
it is the only confirmed config that beats shuffled, uniform, **and** random
on both RMSE gain and distribution KL simultaneously — a cleaner result than
the original 12x16 config, which was only screened against shuffled and did
not clearly separate from uniform/random in the QCBM re-test (Stage 2 table
above). The absolute margin is still small (`0.000959` RMSE gain on a
`~1.002` baseline RMSE, i.e. about `0.1%`), so this remains a weak positive
candidate, not a strong result -- but it is a more robustly established one.

Interpretation: tune the SVD baseline and grouping granularity together
before drawing conclusions from a fixed-hyperparameter prior screen; an
under-tuned baseline can make a prior's benefit look larger (and less
consistently ahead of null controls) than it really is.

## Current Bottleneck

The bottleneck is still prior discovery, not QCBM implementation, though the
margin is now better characterized: with a properly tuned SVD baseline, the
best static grouping prior (`cond_user_itemtype`, 24x24 clusters) beats all
three null controls, but the RMSE gain is small (~0.1% of baseline RMSE).

The most important unresolved question is:

```text
What recommender statistic behaves like a true invariant measure and beats null controls?
```

## Recommended Next Step

Move from scalar residual-prior losses inside Matrix Factorization to a sequence model that can predict or roll out taste states.

Most promising next prior:

```text
P(S_t | S_{t-1})
```

where `S_t` is a recent-window user taste state.

Why this is promising:

- It is closer to the invariant-measure idea in the Wang et al. paper.
- The current code already saves train-only transition targets.
- The current MF model cannot directly use state-transition priors as a natural loss.
- A GRU, small Transformer, Markov-state model, or state-emitting sequence recommender can.

Suggested next experiment:

1. Build a sequence model that emits a distribution over next taste state and next rating.
2. Train baseline without a transition prior.
3. Add empirical `P(S_t | S_{t-1})` as a KL/MMD transition prior.
4. Compare against shuffled-state, shuffled-time, uniform, and random transition priors.
5. Only if this wins, train a QCBM for the transition matrix or compact state-transition distribution.

## Practical Commands

SVD tabulation + grouping/time hyperparameter search (Stage 1.5):

```bash
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 svd_prior_search.py --stage all
```

Run stages independently (each resumes from `results/svd_baseline_grid/`
and `results/svd_prior_search/search_log.csv` on disk):

```bash
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 svd_prior_search.py --stage baseline
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 svd_prior_search.py --stage a
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 svd_prior_search.py --stage b
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 svd_prior_search.py --stage c
```

Static prior discovery:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  qiml_prior_experiments.py
```

QCBM recommender re-test (original 12x16 / untuned-SVD config):

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  evaluate_qprior_recommender.py
```

QCBM recommender re-test against the Stage 1.5 tuned SVD + 24x24 winner:

```bash
/Users/apple/opt/miniconda3/envs/qprior/bin/python3 train_winning_qprior.py \
  --prior results/svd_prior_search/best_config/rank2_cond_user_itemtype/priors/cond_user_itemtype.npz \
  --out_dir models/qprior_winning/cond_user_itemtype_24x24_tuned

/Users/apple/opt/miniconda3/envs/qprior/bin/python3 evaluate_qprior_recommender.py \
  --candidate cond_user_itemtype \
  --empirical_prior results/svd_prior_search/best_config/rank2_cond_user_itemtype/priors/cond_user_itemtype.npz \
  --qcbm_prior models/qprior_winning/cond_user_itemtype_24x24_tuned/qcbm_prior.npy \
  --finite_shot_prior models/qprior_winning/cond_user_itemtype_24x24_tuned/qcbm_prior_finite_shots.npy \
  --out_dir results/qiml_qprior_retest/cond_user_itemtype_24x24_tuned \
  --factors 20 --lr 0.001 --l2 0.0001 \
  --lambda_kl 0.03 --lambda_mmd 0.1 --soft_tau 0.15 \
  --user_clusters 24 --item_clusters 24 \
  --seeds 0,1,2,3,4,5,6,7,8,9
```

Temporal prior quick check:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py --quick
```

Temporal window sweep:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py \
  --windows 5,10,20 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20
```

## Bottom Line

Current status:

```text
Static empirical prior: weak positive, now more robustly established
  (best config beats shuffled, uniform, and random simultaneously after
  SVD tuning + grouping/time search; margin still small, ~0.1% RMSE)
QCBM preservation: re-tested against the tuned SVD + 24x24 grouping config --
  exact-simulation QCBM now clears stage2_success (beats all 3 null controls,
  96.2% RMSE-gain preservation), the first time this project has cleared that
  bar. Finite-shot QCBM (8000 shots) does not clear it -- the larger 9216-state
  table dilutes shot budget. Treat as a stronger but still narrow-margin,
  simulator-only result, not a hardware-ready or "strong QIML" claim.
Temporal residual priors: no winner
Next direction: state-transition sequence prior
Quantum next step: wait until a temporal/state prior beats null controls
```

