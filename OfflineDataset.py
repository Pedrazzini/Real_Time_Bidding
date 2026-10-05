import numpy as np

def sigmoid(w):
    return 1.0 / (1.0 + np.exp(-w))

def generate_task_action(rng, T=1000, d_x=5, d_z=2):
    """
    Setting SYNTHETIC del paper (Appendix B.1.1), un singolo task-azione:
      Z ~ N(0_2, I_2)                      (informazione a priori dell'azione)
      X_t ~ N(0_5, I_5) i.i.d.             (contesti)
      Latenti (NON osservati dal modello):
        U_const ~ N(0, 1)
        U_Z     ~ N(1_2, 0.25^2 I_2)
        U_X     ~ N(1_5, 0.25^2 I_5)
        U_cross = diag(u_1, u_2), u_k ~ N(1, 0.25^2)
      W_t = U_const + U_Z^T Z + U_X^T X_t + X_{t,1:2}^T U_cross Z
      Y_t | W_t ~ Bernoulli(sigmoid(W_t))
    """
    Z = rng.normal(size=d_z)
    X = rng.normal(size=(T, d_x))

    U_const = rng.normal(0.0, 1.0)
    U_Z = rng.normal(1.0, 0.25, size=d_z)
    U_X = rng.normal(1.0, 0.25, size=d_x)
    u_cross = rng.normal(1.0, 0.25, size=2)          # diagonale di U_cross

    W = (U_const
         + U_Z @ Z
         + X @ U_X
         + (X[:, :2] * u_cross * Z[:2]).sum(axis=1))  # X_{t,1:2}^T diag(u) Z_{1:2}
    probs = sigmoid(W)
    Y = rng.binomial(1, probs)

    return {"Z": Z, "X": X, "Y": Y, "probs": probs,
            "U": {"const": U_const, "Z": U_Z, "X": U_X, "cross": u_cross}}


def generate_offline_dataset(n_tasks, T=1000, d_x=5, d_z=2, seed=0):
    rng = np.random.default_rng(seed)
    return [generate_task_action(rng, T=T, d_x=d_x, d_z=d_z) for _ in range(n_tasks)]


# Iperparametri (Appendix B.2.3 del paper)
N_TASKS = 20_000
T = 1000
D_X = 5
D_Z = 2
SEED = 0

offline_data = generate_offline_dataset(n_tasks=N_TASKS, T=T, d_x=D_X, d_z=D_Z, seed=SEED)

n_train = 10_000
train_data = offline_data[:n_train]
val_data = offline_data[n_train:]

print(f"Task-action totali: {len(offline_data)}")
print(f"Train: {len(train_data)} | Validation: {len(val_data)}")
print(f"Shape esempio -> Z: {train_data[0]['Z'].shape}, "
      f"X: {train_data[0]['X'].shape}, Y: {train_data[0]['Y'].shape}")
print(f"Frazione media di click (train): {np.mean([d['Y'].mean() for d in train_data]):.3f}")
print(f"Esempio di elemento nel dataset: {offline_data[3]}")

# Riferimento: loss di un modello che conosce i parametri latenti U
p = np.concatenate([d["probs"] for d in val_data])
floor = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
print(f"Loss minima con U noto (irraggiungibile da p_theta): {floor:.4f}")

import pickle
from pathlib import Path

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "offline_dataset_paper.pkl", "wb") as f:
    pickle.dump(
        {"train_data": train_data, "val_data": val_data,
         "config": {"n_tasks": N_TASKS, "T": T, "d_x": D_X, "d_z": D_Z, "seed": SEED}},
        f,
    )
print(f"Dataset salvato in: {output_dir / 'offline_dataset_paper.pkl'}")        
