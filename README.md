# Q-Prior

Prior-first quantum-inspired machine learning experiments for MovieLens recommendation priors.

## Setup

This project was prepared from the local conda environment named `qprior`.

```bash
conda create -n qprior python=3.12
conda activate qprior
pip install -r requirements.txt
```

## Data

Dataset files are intentionally excluded from this repository. The experiments expect the MovieLens 100K dataset under:

```text
data/ml-100k/
```

The main rating file should be available as:

```text
data/ml-100k/u.data
```

## Experiments

See `README_QIML_EXPERIMENTS.md` for the experiment pipeline, commands, and output descriptions.
