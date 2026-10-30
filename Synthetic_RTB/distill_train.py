"""
Allenamento della rete STUDENTE che sostituisce il fitting dell'MLP-policy nella fase 2 di TS-Gen.

Input  (47) : [x_q standardizzato (5), feature standardizzate del dataset imputato (42)]
Target      : P(Y=1 | x_q) prodotta dall'insegnante (MLPClassifier fittato da zero), probabilita' "soft".
Loss        : cross-entropy con target soft (equivale a minimizzare la KL dall'insegnante).

Lo split train/validation e' fatto PER TASK (i dataset dello stesso task sono molto correlati).
I dataset "costanti" (tutti gli Y imputati uguali) sono esclusi dal training: l'agente li gestisce
senza rete, restituendo la media, come fa l'insegnante.

Salva "policy_student.pt" con le chiavi lette da TSGenDistilledAgent:
state_dict, d_in, hidden, F_mean, F_std, Q_mean, Q_std.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------------------------------
DATA_PATH = "distill_data/distill_dataset.npz"
SAVE_PATH = "policy_student.pt"
HIDDEN = 128
N_EPOCHS = 80
BATCH_FITS = 256          # numero di dataset (MLP fittati) per batch; ogni dataset ha K contesti
LR = 2e-3
WEIGHT_DECAY = 1e-4
VAL_TASK_FRACTION = 0.15
SEED = 0
device = "cpu"            # la rete e' minuscola: la CPU basta


# ----------------------------------------------------------------------------------------------
# MODELLO (copiare identica nello script delle aste)
# ----------------------------------------------------------------------------------------------
class PolicyStudent(nn.Module):
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)           # logit


def make_inputs(f, q):
    """f: (B, d_feat) feature del dataset; q: (B, K, d_x) contesti. Ordine: [q, f] (come nell'agente)."""
    B, K, _ = q.shape
    return torch.cat([q, f[:, None, :].expand(B, K, f.shape[1])], dim=2).reshape(B * K, -1)


def predict_fits(model, F, Q, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(F), bs):
            f, q = F[i:i + bs].to(device), Q[i:i + bs].to(device)
            out.append(torch.sigmoid(model(make_inputs(f, q))).reshape(f.shape[0], -1).cpu())
    return torch.cat(out).numpy()


def metrics(p_pred, p_true):
    eps = 1e-7
    s = np.clip(p_pred, eps, 1 - eps)
    p = np.clip(p_true, eps, 1 - eps)
    bce = -np.mean(p * np.log(s) + (1 - p) * np.log(1 - s))
    ent = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
    return {"BCE": bce, "KL": bce - ent, "MAE": np.mean(np.abs(p_pred - p_true)),
            "RMSE": np.sqrt(np.mean((p_pred - p_true) ** 2))}


def fmt(d):
    return " | ".join(f"{k} {v:.4f}" for k, v in d.items())


# ----------------------------------------------------------------------------------------------
# DATI
# ----------------------------------------------------------------------------------------------
data = np.load(DATA_PATH)
F_raw, Q_raw, P_all, meta = data["F"], data["Q"], data["P"], data["meta"]
# meta: task, t, azione, n_righe, costante
n_fits, K, d_x = Q_raw.shape
d_feat = F_raw.shape[1]
d_in = d_x + d_feat
is_const = meta[:, 4].astype(bool)
print(f"MLP fittati: {n_fits} | contesti per dataset K={K} | dataset costanti: {is_const.mean():.3f}")

rng = np.random.default_rng(SEED)
tasks = np.unique(meta[:, 0])
val_tasks = set(rng.choice(tasks, size=max(1, int(VAL_TASK_FRACTION * len(tasks))), replace=False).tolist())
in_val = np.array([m in val_tasks for m in meta[:, 0]])

tr_idx = np.flatnonzero(~in_val & ~is_const)
va_all = np.flatnonzero(in_val)                  # ordine originale (serve per l'accordo sulle decisioni)
va_nc = np.flatnonzero(in_val & ~is_const)
print(f"Task train/val: {len(tasks) - len(val_tasks)}/{len(val_tasks)} | "
      f"dataset train: {len(tr_idx)} | val (non costanti): {len(va_nc)}")

# standardizzazione calcolata SOLO sul train
F_mean = F_raw[tr_idx].mean(axis=0)
F_std = F_raw[tr_idx].std(axis=0) + 1e-6
Q_mean = Q_raw[tr_idx].reshape(-1, d_x).mean(axis=0)
Q_std = Q_raw[tr_idx].reshape(-1, d_x).std(axis=0) + 1e-6

F_n = torch.from_numpy(((F_raw - F_mean) / F_std).astype(np.float32))
Q_n = torch.from_numpy(((Q_raw - Q_mean) / Q_std).astype(np.float32))
P_t = torch.from_numpy(P_all.astype(np.float32))


# ----------------------------------------------------------------------------------------------
# BASELINE BANALE: prevedere la media degli Y imputati del dataset (feature n.1)
# ----------------------------------------------------------------------------------------------
base_pred = np.repeat(F_raw[va_nc, 1][:, None], K, axis=1)
print("\nBaseline 'media di y'   (val):", fmt(metrics(base_pred, P_all[va_nc])))


# ----------------------------------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------------------------------
torch.manual_seed(SEED)
model = PolicyStudent(d_in, HIDDEN).to(device)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
loss_fn = nn.BCEWithLogitsLoss()                 # accetta target soft in [0,1]

best_val, best_state = float("inf"), None
t_start = time.time()
for epoch in range(N_EPOCHS):
    model.train()
    perm = rng.permutation(tr_idx)
    tot, nb = 0.0, 0
    for i in range(0, len(perm), BATCH_FITS):
        b = torch.from_numpy(perm[i:i + BATCH_FITS])
        f, q, p = F_n[b].to(device), Q_n[b].to(device), P_t[b].to(device)
        opt.zero_grad()
        loss = loss_fn(model(make_inputs(f, q)), p.reshape(-1))
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    sched.step()

    val_pred = predict_fits(model, F_n[va_nc], Q_n[va_nc])
    m = metrics(val_pred, P_all[va_nc])
    if m["BCE"] < best_val:
        best_val = m["BCE"]
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
        print(f"Epoch {epoch + 1:3d}/{N_EPOCHS} - train BCE {tot / nb:.4f} - val: {fmt(m)}")
print(f"\nTraining concluso in {time.time() - t_start:.0f}s")

model.load_state_dict(best_state)


# ----------------------------------------------------------------------------------------------
# VALUTAZIONE FINALE
# ----------------------------------------------------------------------------------------------
val_pred_nc = predict_fits(model, F_n[va_nc], Q_n[va_nc])
print("\nStudente (miglior epoca) (val):", fmt(metrics(val_pred_nc, P_all[va_nc])))
print("Baseline 'media di y'    (val):", fmt(metrics(base_pred, P_all[va_nc])))

# Accordo sulle DECISIONI: a parita' di (task, t), argmax tra le azioni (contesto 0 = vero X_t),
# prezzi uguali. Per i dataset costanti si usa il valore dell'insegnante (l'agente fa lo stesso).
n_actions = int(meta[:, 2].max()) + 1
pred_all = predict_fits(model, F_n[va_all], Q_n[va_all])
pred_all[is_const[va_all]] = P_all[va_all][is_const[va_all]]
mv = meta[va_all]
ok = (len(va_all) % n_actions == 0)
if ok:
    g_task = mv[:, 0].reshape(-1, n_actions)
    g_t = mv[:, 1].reshape(-1, n_actions)
    g_a = mv[:, 2].reshape(-1, n_actions)
    ok = (g_task == g_task[:, :1]).all() and (g_t == g_t[:, :1]).all() and \
         (g_a == np.arange(n_actions)).all()
if ok:
    teach = P_all[va_all][:, 0].reshape(-1, n_actions)
    stud = pred_all[:, 0].reshape(-1, n_actions)
    agree = (teach.argmax(axis=1) == stud.argmax(axis=1)).mean()
    print(f"Accordo sull'azione scelta (argmax, prezzi uguali) su {teach.shape[0]} istanti: {agree:.3f}")
else:
    print("Accordo sulle decisioni non calcolabile: ordine dei gruppi (task, t, azione) non regolare.")

# Tempo di inferenza: un forward su n_actions righe (quello che fara' l'agente)
x = torch.randn(n_actions, d_in)
model.eval()
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(1000):
        model(x)
    dt = (time.perf_counter() - t0) / 1000
print(f"Tempo di inferenza: {dt * 1e3:.3f} ms per decisione (insegnante: ~secondi)")

# ----------------------------------------------------------------------------------------------
# SALVATAGGIO (solo tipi semplici: si carica anche con torch.load(weights_only=True))
# ----------------------------------------------------------------------------------------------
torch.save({
    "state_dict": model.state_dict(),
    "d_in": d_in,
    "hidden": HIDDEN,
    "F_mean": F_mean.astype(float).tolist(),
    "F_std": F_std.astype(float).tolist(),
    "Q_mean": Q_mean.astype(float).tolist(),
    "Q_std": Q_std.astype(float).tolist(),
}, SAVE_PATH)
print(f"Salvato in {SAVE_PATH}")