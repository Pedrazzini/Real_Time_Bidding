import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def bootstrap_task_action(rng, X, Y, T=None):
    """
    Algorithm 4 (Appendix B.2.2): ricampiona con reinserimento T coppie (X,Y)
    dal pool storico S^(a) di un task-azione, per formare tilde{tau}^(a).
    Nel nostro caso il pool è l'intero dataset generato per quel task-azione.
    """
    n_pool = X.shape[0]
    if T is None:
        T = n_pool
    idx = rng.integers(0, n_pool, size=T)
    return X[idx], Y[idx]



def build_summary_stats(X, Y, ridge=1.0):
    """
    Per ogni t = 1,...,T calcola, usando SOLO la storia s < t:
        - (X^T X + I)^{-1}   (d_x x d_x)
        - X^T Y              (d_x,)
    Vettorizzato con somme cumulative "esclusive" + inversione batch di matrici 5x5.
    Per t=1 la storia è vuota -> (X^T X + I)^{-1} = I, X^T Y = 0.
    """
    T, d_x = X.shape

    outer = np.einsum('ti,tj->tij', X, X)              # (T, d_x, d_x)
    cum_outer = np.cumsum(outer, axis=0)  #---> per ora ho una lista di matrici 5x5 dove la prima è X-1^(T)X_1 (ove la matrice X_1 è una matrice 1x5), la seconda è X_2^(T)X_2 (ove la matrice X_2 è una matrice 2x5)... ecc
    cum_outer_excl = np.concatenate(
        [np.zeros((1, d_x, d_x)), cum_outer[:-1]], axis=0
    )

    cum_XY = np.cumsum(X * Y[:, None], axis=0)
    cum_XY_excl = np.concatenate(
        [np.zeros((1, d_x)), cum_XY[:-1]], axis=0
    )

    S = cum_outer_excl + ridge * np.eye(d_x)[None, :, :]
    S_inv = np.linalg.inv(S)                            # (T, d_x, d_x), batch invert

    return S_inv, cum_XY_excl


def build_training_examples(Z, X, Y, repeat_factor=100, ridge=1.0):
    """
    Input per il timestep t (Figure 7 / Appendix B.2.1):
        [Z, X_t, repeat( vec((X^T X + I)^-1) concat (X^T Y), repeat_factor volte )]
    Target: Y_t
    Costruito per tutti i t = 1,...,T in un colpo solo (vettorizzato).
    """
    T, d_x = X.shape
    S_inv, XY = build_summary_stats(X, Y, ridge=ridge)

    S_inv_flat = S_inv.reshape(T, -1)                       # (T, d_x*d_x)
    summary = np.concatenate([S_inv_flat, XY], axis=1)      # (T, d_x*d_x + d_x)
    summary_repeated = np.tile(summary, (1, repeat_factor))  # (T, repeat_factor*(d_x*d_x+d_x))

    Z_repeated = np.tile(Z, (T, 1))                          # (T, d_z)

    inputs = np.concatenate([Z_repeated, X, summary_repeated], axis=1)
    targets = Y.astype(np.float32)

    return inputs.astype(np.float32), targets



class SequenceModelMLP(nn.Module):
    """
    Input layer -> 3 hidden layer (width 100, ReLU) -> output layer -> logit.
    Sigmoide applicata implicitamente da BCEWithLogitsLoss in training.
    """
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits, shape (batch,)
    

def make_batch(rng, tasks, batch_size, repeat_factor=100, ridge=1.0, bootstrap=True):
    chosen_idx = rng.choice(len(tasks), size=batch_size, replace=False)
    all_inputs, all_targets = [], []

    for i in chosen_idx:
        task = tasks[i]
        Z, X, Y = task["Z"], task["X"], task["Y"]
        if bootstrap:
            X, Y = bootstrap_task_action(rng, X, Y)
        inputs, targets = build_training_examples(Z, X, Y, repeat_factor, ridge)
        all_inputs.append(inputs)
        all_targets.append(targets)

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def train_p_theta(train_data, val_data, n_epochs=100, batch_size=500,
                   lr=0.01, weight_decay=0.01, repeat_factor=100, seed=0,
                   device="cpu"):
    rng = np.random.default_rng(seed)
    d_z = train_data[0]["Z"].shape[0]
    d_x = train_data[0]["X"].shape[1]
    input_dim = d_z + d_x + repeat_factor * (d_x * d_x + d_x)

    model = SequenceModelMLP(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n_batches_per_epoch = max(1, len(train_data) // batch_size)
    history = {"train_loss": [], "val_loss": []}

    best_val, best_state = float("inf"), None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            inputs, targets = make_batch(rng, train_data, batch_size, repeat_factor)
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / n_batches_per_epoch

        model.eval()
        with torch.no_grad():
            val_inputs, val_targets = make_batch(
                rng, val_data, min(batch_size, len(val_data)),
                repeat_factor, bootstrap=False
            )
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_loss = loss_fn(model(val_inputs), val_targets).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - train loss: {train_loss:.4f} - val loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history

from pathlib import Path
import pickle

# Ridefinisci la variabile nel nuovo notebook
output_dir = Path("data")

with open(output_dir / "offline_dataset.pkl", "rb") as f:
    loaded = pickle.load(f)

train_data = loaded["train_data"]
val_data = loaded["val_data"]
config = loaded["config"]

print("CUDA available?:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will run on: {device}")

device = "cpu" # sovrascrivo di nuovo il device e lo imposto come CPU, perchè ha più RAM (anche se lenta) della GPU e riesce a processare tutti i dati

model, history = train_p_theta(
    train_data, val_data,
    n_epochs=50, batch_size=500, # provo con 50 epoche anche se nel paper lo faceva con 100
    device=device   # <-- passa qui il device rilevato
)

