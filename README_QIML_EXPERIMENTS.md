# Prior-First QIML Experiments

This workspace now has a script-based experiment pipeline for testing which
recommendation prior is meaningful before training a quantum circuit.

## Stage 1: classical prior discovery

Run a smoke test:

```bash
python3 qiml_prior_experiments.py --quick
```

If the shell `python3` does not have `numpy` and `pandas`, use the Codex
bundled runtime available in this workspace:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  qiml_prior_experiments.py --quick
```

Run the planned MovieLens 100K campaign:

```bash
python3 qiml_prior_experiments.py \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20 \
  --lambda_kl 0.01 \
  --lambda_mmd 0.05
```

Outputs are written to `results/qiml_prior_experiments/`:

- `metrics.csv`: rating, ranking, and distribution metrics for every run.
- `history.csv`: per-epoch losses and validation metrics.
- `summary.csv`: mean/stdev by prior candidate and variant.
- `winner_screen.csv`: paired screen for priors that beat shuffled controls.
- `prior_catalog.csv`: saved prior shapes and descriptions.
- `priors/*.npz`: empirical priors usable by the QCBM stage.

The runner compares each candidate against:

- baseline MF
- MF plus real prior
- MF plus shuffled prior
- MF plus uniform prior
- MF plus random prior

The implemented candidates are:

- `global_rating`: `P(r)`
- `global_residual`: `P(E)`
- `global_dot`: `P(p_u dot q_i)`, built from the seed's baseline MF
- `cond_activity_popularity`: `P(E | user_activity_bin, item_popularity_bin)`
- `cond_user_genre`: `P(E | residual_lift_user_genre_type, primary_genre)`
- `cond_user_itemtype`: `P(E | residual_user_cluster, item_cluster)`

The loss uses KL plus MMD-style L2 distribution matching. This replaces the
notebook's pure cross-entropy residual prior so the diagnostic measures true
distribution alignment instead of only mode-seeking behavior.

## Stage 2: QCBM for a winning prior

Only run this after `winner_screen.csv` identifies a meaningful prior.

```bash
python3 train_winning_qprior.py \
  --prior results/qiml_prior_experiments/priors/cond_user_itemtype.npz \
  --out_dir models/qprior_winning/cond_user_itemtype
```

This script requires `torch` and `pennylane`. It trains a frozen QCBM prior,
writes the learned distribution, adds a finite-shot version, and reports
compression against the empirical prior table.

After the QCBM prior is trained, close the loop by re-testing the recommender:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  evaluate_qprior_recommender.py
```

This compares baseline MF, empirical prior, QCBM prior, finite-shot QCBM prior,
and shuffled/uniform/random controls. Outputs are written to
`results/qiml_qprior_retest/cond_user_itemtype/`:

- `summary.csv`: mean metrics for each variant.
- `preservation.csv`: QCBM gain divided by empirical-prior gain.
- `qprior_fit.csv`: QCBM-vs-empirical distribution fit after reshaping.
- `metrics_with_gains.csv`: per-seed metrics and gain columns.

## Interpreting success

A prior is a real candidate only if it improves heldout statistical fidelity,
does not degrade RMSE beyond the configured guardrail, and beats shuffled/random
controls across paired seeds. If no prior survives Stage 1, the correct
conclusion is that the current recommender adaptation has not yet found a useful
analogue of the paper's invariant measure.

## Temporal prior experiments

The temporal runner treats each user's ratings as a sequence and tests priors
conditioned on recent taste state:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py --quick
```

Full default run:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py \
  --window 10 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20
```

Window sweep for the planned `L in [5, 10, 20]` comparison:

```bash
/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  temporal_qiml_experiments.py \
  --windows 5,10,20 \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --epochs 20
```

Outputs are written to `results/temporal_qiml_prior_experiments/`.

Implemented temporal priors:

- `temporal_next_residual`: `P(E_t | S_{t-1}, item_type_t)`
- `temporal_delta_residual`: `P(delta E_t | E_{t-1 bin}, S_{t-1}, item_type_t)`
- `temporal_gap_residual`: `P(E_t | gap_bucket, S_{t-1}, item_type_t)`
- `temporal_global_residual`: global temporal residual control

Each prior is compared against real, shuffled-user, shuffled-time, uniform, and
random variants. The runner also reports residual autocorrelation error,
state-occupancy KL, and state-transition KL as temporal analogues of the
paper's invariant-measure checks. Train-only diagnostic targets are saved in
`diagnostic_priors/`:

- `state_occupancy.npz`
- `state_transition.npz`
- `residual_autocorr.npz`
