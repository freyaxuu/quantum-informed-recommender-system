#!/usr/bin/env python3
"""Train a PennyLane QCBM for a saved winning prior distribution.

Stage 1 should first identify a meaningful prior and save it as an `.npz`
under `results/qiml_prior_experiments/priors/`.  This script fits a compact
QCBM to that prior, freezes the learned distribution, and reports how closely
the circuit approximates the empirical prior.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def normalize(x, eps=1e-10):
    x = np.maximum(np.asarray(x, dtype=np.float64), eps)
    return x / x.sum()


def flatten_prior(path: Path):
    data = np.load(path, allow_pickle=True)
    centers = data["centers"].astype(np.float64)
    probs = data["probs"].astype(np.float64)
    flat = normalize(probs.reshape(-1))
    meta = {
        "source_prior": str(path),
        "source_shape": list(probs.shape),
        "centers": centers.tolist(),
        "value_kind": str(data["value_kind"]),
        "mode": str(data["mode"]),
        "name": str(data["name"]),
        "description": str(data["description"]) if "description" in data else "",
    }
    return flat, meta


def fit_qcbm(target, n_qubits, layers, epochs, lr, seed):
    try:
        import pennylane as qml
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(
            "PennyLane and torch are required for Stage 2. Install them in the "
            "environment used to run this script, or run Stage 1 only."
        ) from exc

    n_states = 2**n_qubits
    if len(target) > n_states:
        raise ValueError(f"Prior has {len(target)} states but {n_qubits} qubits provide only {n_states}.")
    padded = np.zeros(n_states, dtype=np.float64)
    padded[: len(target)] = target
    padded = normalize(padded)

    torch.manual_seed(seed)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(params):
        for layer in range(layers):
            for wire in range(n_qubits):
                qml.RX(params[layer, wire, 0], wires=wire)
                qml.RY(params[layer, wire, 1], wires=wire)
                qml.RZ(params[layer, wire, 2], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.probs(wires=range(n_qubits))

    class QCBM(nn.Module):
        def __init__(self):
            super().__init__()
            self.params = nn.Parameter(0.01 * torch.randn(layers, n_qubits, 3, dtype=torch.float64))

        def forward(self):
            probs = circuit(self.params)
            useful = probs[: len(target)]
            return useful / useful.sum()

    target_t = torch.tensor(target, dtype=torch.float64)
    model = QCBM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        pred = model()
        eps = 1e-10
        kl = torch.sum(pred * torch.log((pred + eps) / (target_t + eps)))
        mmd_l2 = torch.sum((pred - target_t) ** 2)
        loss = kl + mmd_l2
        loss.backward()
        opt.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "kl": float(kl.detach()), "mmd_l2": float(mmd_l2.detach())})
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch {epoch:04d} loss={history[-1]['loss']:.8f} kl={history[-1]['kl']:.8f}")

    learned = model().detach().numpy()
    state_dict = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    return learned, state_dict, history


def finite_shot_distribution(probs, shots, seed):
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, normalize(probs))
    return normalize(counts + 1e-10)


def dist_metrics(p, q):
    p = normalize(p)
    q = normalize(q)
    eps = 1e-10
    m = 0.5 * (p + q)
    return {
        "kl": float(np.sum(p * np.log((p + eps) / (q + eps)))),
        "js": float(0.5 * np.sum(p * np.log((p + eps) / (m + eps))) + 0.5 * np.sum(q * np.log((q + eps) / (m + eps)))),
        "mmd_l2": float(np.sum((p - q) ** 2)),
        "max_abs_error": float(np.max(np.abs(p - q))),
    }


def main():
    parser = argparse.ArgumentParser(description="Train QCBM for a selected prior.")
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("models/qprior_winning"))
    parser.add_argument("--n_qubits", type=int, default=None)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots", type=int, default=8000)
    args = parser.parse_args()

    target, meta = flatten_prior(args.prior)
    n_qubits = args.n_qubits or math.ceil(math.log2(len(target)))
    learned, state_dict, history = fit_qcbm(target, n_qubits, args.layers, args.epochs, args.lr, args.seed)
    shot_dist = finite_shot_distribution(learned, args.shots, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "target_prior.npy", target)
    np.save(args.out_dir / "qcbm_prior.npy", learned)
    np.save(args.out_dir / "qcbm_prior_finite_shots.npy", shot_dist)
    np.savez_compressed(args.out_dir / "qcbm_state.npz", **state_dict)

    report = {
        **meta,
        "n_states": len(target),
        "n_qubits": n_qubits,
        "layers": args.layers,
        "trainable_parameters": args.layers * n_qubits * 3,
        "shots": args.shots,
        "qcbm_vs_empirical": dist_metrics(learned, target),
        "finite_shot_vs_empirical": dist_metrics(shot_dist, target),
        "history": history,
        "compression": {
            "empirical_prior_bytes": int(target.nbytes),
            "qcbm_parameter_bytes_float64": int(args.layers * n_qubits * 3 * 8),
        },
    }
    (args.out_dir / "qcbm_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote QCBM prior artifacts to {args.out_dir}")
    print(json.dumps({k: report[k] for k in ["qcbm_vs_empirical", "finite_shot_vs_empirical", "compression"]}, indent=2))


if __name__ == "__main__":
    main()
