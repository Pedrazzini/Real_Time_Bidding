import numpy as np

def generate_task_action(rng, T=1000, d_x=5, d_z=5, temperature=None):
    """
    Genera un singolo dataset "task-action" (coerente con l'assunzione di
    indipendenza tra azioni usata nel setting sintetico del paper):
    
    - Z ~ N(0_5, I_5): parametro latente dell'azione (5 dim, come X)
    - X_1, ..., X_T ~ N(0_5, I_5) i.i.d.: contesti
    - Y_t | X_t, Z ~ Bernoulli( sigmoid( (X_t^T Z) / temperature ) )
    
    Il fattore 'temperature' scala il prodotto scalare per evitare probabilità
    troppo estreme: dato che X_t, Z ~ N(0, I_d) indipendenti, Var(X_t^T Z) = d,
    quindi usiamo temperature = sqrt(d_x) come scala di default.
    """
    if temperature is None:
        temperature = np.sqrt(d_x)

    Z = rng.normal(size=d_z)
    X = rng.normal(size=(T, d_x))

    logits = (X @ Z) / temperature
    probs = 1.0 / (1.0 + np.exp(-logits))
    Y = rng.binomial(1, probs)

    return {"Z": Z, "X": X, "Y": Y, "probs": probs}


def generate_offline_dataset(n_tasks, T=1000, d_x=5, d_z=5, temperature=None, seed=0):
    """
    Genera n_tasks dataset "task-action" indipendenti, da usare come D_offline
    per il training di p_theta (Sezione 4.1 / Appendix B.2.3 del paper).
    """
    rng = np.random.default_rng(seed)
    dataset = [
        generate_task_action(rng, T=T, d_x=d_x, d_z=d_z, temperature=temperature)
        for _ in range(n_tasks)
    ]
    return dataset

# Iperparametri (in linea con Appendix B.2.3 del paper)
N_TASKS = 20_000
T = 1000
D_X = 5
D_Z = 5
SEED = 0

offline_data = generate_offline_dataset(
    n_tasks=N_TASKS, T=T, d_x=D_X, d_z=D_Z, seed=SEED
)

n_train = 10_000
train_data = offline_data[:n_train]
val_data = offline_data[n_train:]

print(f"Task-action totali: {len(offline_data)}")
print(f"Train: {len(train_data)} | Validation: {len(val_data)}")
print(f"Shape esempio -> Z: {train_data[0]['Z'].shape}, "
      f"X: {train_data[0]['X'].shape}, Y: {train_data[0]['Y'].shape}")
print(f"Frazione media di click (train): {np.mean([d['Y'].mean() for d in train_data]):.3f}")
print(f"esempio di elemento: {offline_data[2]}" )


import pickle
from pathlib import Path

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "offline_dataset.pkl", "wb") as f:
    pickle.dump(
        {
            "train_data": train_data,
            "val_data": val_data,
            "config": {
                "n_tasks": N_TASKS,
                "T": T,
                "d_x": D_X,
                "d_z": D_Z,
                "seed": SEED,
            },
        },
        f,
    )

print(f"Dataset salvato in: {output_dir / 'offline_dataset.pkl'}")