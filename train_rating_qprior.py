import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import argparse
import os
import time

# --------------------------------------------------
# 0. Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(
    description="Train Q-Prior on recommender rating distribution"
)
parser.add_argument("--data_path", type=str, default="../data/ml-100k/u.data")
parser.add_argument("--n_qubits", type=int, default=3)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--num_layers", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="../models/qprior_movielens")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

n_qubits = args.n_qubits
num_states = 2 ** n_qubits
num_ratings = 5

assert num_states >= num_ratings, "Need at least 3 qubits for 5 rating bins."

# --------------------------------------------------
# 1. Load MovieLens ratings
# --------------------------------------------------
# u.data format:
# user_id | item_id | rating | timestamp
data = np.loadtxt(args.data_path, delimiter="\t", dtype=np.int64)

users = data[:, 0]
items = data[:, 1]
ratings = data[:, 2]

print(f"Loaded {len(ratings)} ratings")
print(f"Users: {users.min()} to {users.max()}")
print(f"Items: {items.min()} to {items.max()}")
print(f"Rating values: {np.unique(ratings)}")

# Empirical distribution over ratings 1,2,3,4,5
counts = np.array([(ratings == r).sum() for r in range(1, 6)], dtype=np.float32)
empirical_prior = counts / counts.sum()

print("Empirical rating distribution:")
for r, p in zip(range(1, 6), empirical_prior):
    print(f"Rating {r}: {p:.4f}")

target_probs = torch.tensor(empirical_prior, dtype=torch.float32)

# --------------------------------------------------
# 2. Quantum Generator
# --------------------------------------------------
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def QG_circuit(params):
    for l in range(args.num_layers):
        for i in range(n_qubits):
            qml.RY(params[l, i, 0], wires=i)
            qml.RZ(params[l, i, 1], wires=i)
            qml.RX(params[l, i, 2], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return qml.probs(wires=range(n_qubits))


class QPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(
            0.01 * torch.randn(args.num_layers, n_qubits, 3)
        )

    def forward(self):
        probs = QG_circuit(self.params)

        # Use first 5 quantum basis states as rating bins 1-5.
        rating_probs = probs[:num_ratings]

        # Renormalize because remaining quantum states are unused.
        rating_probs = rating_probs / rating_probs.sum()

        return rating_probs


# --------------------------------------------------
# 3. Loss: MMD-style / L2 distribution matching
# --------------------------------------------------
def qprior_loss(pred_probs, target_probs):
    return torch.sum((pred_probs - target_probs) ** 2)


# --------------------------------------------------
# 4. Train Q-Prior
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using torch device: {device}")

model = QPrior().to(device)
target_probs = target_probs.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

loss_history = []

start = time.time()

for epoch in range(args.epochs):
    optimizer.zero_grad()

    pred_probs = model()
    loss = qprior_loss(pred_probs, target_probs)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.8f}")

elapsed = time.time() - start
print(f"Training completed in {elapsed:.2f} seconds")

learned_prior = model().detach().cpu().numpy()

print("\nLearned Q-Prior distribution:")
for r, p in zip(range(1, 6), learned_prior):
    print(f"Rating {r}: {p:.4f}")

# --------------------------------------------------
# 5. Save result
# --------------------------------------------------
np.save(os.path.join(args.out_dir, "rating_qprior.npy"), learned_prior)
torch.save(model.state_dict(), os.path.join(args.out_dir, "rating_qprior_model.pt"))

# --------------------------------------------------
# 6. Plot preliminary result
# --------------------------------------------------
ratings_axis = np.arange(1, 6)

plt.figure(figsize=(8, 5))
plt.bar(ratings_axis - 0.15, empirical_prior, width=0.3, label="Empirical")
plt.bar(ratings_axis + 0.15, learned_prior, width=0.3, label="Q-Prior")
plt.xlabel("Rating")
plt.ylabel("Probability")
plt.title("MovieLens Rating Distribution: Empirical vs Q-Prior")
plt.xticks(ratings_axis)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "rating_qprior_prelim.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MMD-style loss")
plt.title("Q-Prior Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "rating_qprior_loss.png"), dpi=300)
plt.show()