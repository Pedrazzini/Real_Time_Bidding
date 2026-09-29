import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def flatten_no_history(tasks):
    """Per ogni task-azione e per ogni t: input = [Z, X_t], target = Y_t."""
    inputs, targets = [], []
    for d in tasks:
        Z, X, Y = d["Z"], d["X"], d["Y"]
        T = X.shape[0]
        inputs.append(np.concatenate([np.tile(Z, (T, 1)), X], axis=1))
        targets.append(Y)
    return (torch.from_numpy(np.concatenate(inputs).astype(np.float32)),
            torch.from_numpy(np.concatenate(targets).astype(np.float32)))

class SmallMLP(nn.Module):
    # stessa struttura del paper (5 layer lineari, width 100), ma input_dim = d_z + d_x
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
        return self.net(x).squeeze(-1)

def diagnostic_no_history(train_data, val_data, n_train_tasks=200, n_val_tasks=200,
                          n_epochs=200, row_batch=5000, lr=1e-4, weight_decay=0.01,
                          seed=0, device="cpu"):
    torch.manual_seed(seed)
    Xtr, Ytr = flatten_no_history(train_data[:n_train_tasks])
    Xva, Yva = flatten_no_history(val_data[:n_val_tasks])
    Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)

    model = SmallMLP(Xtr.shape[1]).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # riferimento teorico: loss di un modello che conosce la probabilità vera
    p = np.concatenate([d["probs"] for d in val_data[:n_val_tasks]])
    floor = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
    print(f"ln 2 = 0.6931 | loss minima teorica (val) = {floor:.4f}")

    N = Xtr.shape[0]
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        tot, nb = 0.0, 0
        for i in range(0, N, row_batch):
            idx = perm[i:i + row_batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()
        print(f"Epoch {epoch+1}/{n_epochs} - train: {tot/nb:.4f} - val: {val_loss:.4f}")
    return model

from pathlib import Path
import pickle

output_dir = Path("data")

with open(output_dir / "offline_dataset_paper.pkl", "rb") as f:
    loaded = pickle.load(f)

train_data = loaded["train_data"]
val_data = loaded["val_data"]
config = loaded["config"]

model_diag = diagnostic_no_history(
    train_data, val_data,
    n_train_tasks=2000, n_val_tasks=1000,
    n_epochs=50, lr=1e-3, device="cpu"
)
