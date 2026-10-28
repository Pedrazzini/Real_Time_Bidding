"""
Costruzione del dataset per DISTILLARE la policy MLP della fase 2 di TS-Gen.

Insegnante: MLPClassifier (sklearn, default) fittato da zero su ogni dataset imputato.
Studente (notebook successivo): rete che riceve un RIASSUNTO del dataset imputato + il contesto
da valutare e restituisce la stessa probabilita' dell'insegnante, senza nessun fitting online.

Un esempio di training e' una coppia (dataset imputato di UNA azione, contesto x_q):
    input  = [x_q (5), feature del dataset (42)]
    label  = P(Y=1 | x_q) prodotta dall'MLP fittato su quel dataset

Per ogni MLP fittato (costoso, ~0.7 s) si etichettano K contesti (quasi gratis): il primo e' il
vero X_t, gli altri sono N(0, I_5) (la distribuzione dei contesti).

Le storie (quali azioni osservate e quando) sono SIMULATE in modo economico: a ogni passo si osserva con
probabilita' q (per task) e si sceglie l'azione con pesi Dirichlet (per task), cosi' si coprono storie
dense/sparse e bilanciate/sbilanciate (nell'asta TS-Gen osserva solo quando vince).
Imputazione e fitting dell'insegnante si fanno solo in TIMES_PER_TASK istanti per task.

Uso:   python build_distillation_dataset.py
Ripartibile: i task gia' salvati in OUT_DIR/tasks vengono saltati.
"""
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

# ----------------------------------------------------------------------------------------------
# CONFIGURAZIONE
# ----------------------------------------------------------------------------------------------
D_X, D_Z = 5, 2
REPEAT_FACTOR = 100
INPUT_DIM = D_Z + D_X + REPEAT_FACTOR * (D_X * D_X + D_X)   # 3007
T = 600                 # come negli esperimenti di asta
HORIZON = 100            # come negli esperimenti di asta
N_ACTIONS = 5
T_REF = 600             # normalizzazione del numero di righe del dataset imputato

N_TASKS = 300          # per una prova: 2
TIMES_PER_TASK = 20    # istanti t in cui si imputa e si fitta l'insegnante; per una prova: 3
K_QUERIES = 40          # contesti etichettati per ogni MLP fittato
Q_OBS_MIN = 0.1         # probabilita' di osservare a ogni passo ~ U(Q_OBS_MIN, 1), per task
BASE_SEED = 100_000     # lontano dai seed degli esperimenti (0..39): niente task in comune col test
N_JOBS = 4              # processi paralleli

OUT_DIR = "distill_data"
P_THETA_PATH = "p_theta.pt"


def sigmoid(w):
    return 1.0 / (1.0 + np.exp(-w))


# ----------------------------------------------------------------------------------------------
# p_theta e utilita' (copiate dallo script delle aste: stesso comportamento)
# ----------------------------------------------------------------------------------------------
class SequenceModelMLP(nn.Module):
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


def load_p_theta(path, device="cpu"):
    m = SequenceModelMLP(INPUT_DIM).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


class RunningStats:
    def __init__(self, d_x, ridge=1.0):
        self.d_x = d_x
        self.ridge = ridge
        self.sum_outer = np.zeros((d_x, d_x))
        self.sum_XY = np.zeros(d_x)
        self.count = 0

    def current(self):
        S = self.sum_outer + self.ridge * np.eye(self.d_x)
        return np.linalg.inv(S), self.sum_XY / max(self.count, 1)

    def update(self, X_i, Y_i):
        self.sum_outer += np.outer(X_i, X_i)
        self.sum_XY += X_i * Y_i
        self.count += 1


def build_query_input(Z, X_i, S_inv, XY_mean, repeat_factor=REPEAT_FACTOR):
    summary = np.concatenate([S_inv.flatten(), XY_mean])
    return np.concatenate([Z, X_i, np.tile(summary, repeat_factor)]).astype(np.float32)


def impute_missing_outcomes(model, Z_per_action, X_hat, observed_times, Y_hist,
                            n_actions, rng, device="cpu"):
    """Algorithm 3 (identica a quella dello script delle aste)."""
    effective_T = X_hat.shape[0]
    Y_hat_per_action = {}
    for a in range(n_actions):
        obs_set = set(observed_times[a])
        missing = [s for s in range(effective_T) if s not in obs_set]

        stats = RunningStats(D_X)
        for s in sorted(obs_set):
            stats.update(X_hat[s], Y_hist[a][s])

        Y_hat_a = dict(Y_hist[a])
        for s in missing:
            S_inv, XY_mean = stats.current()
            inp = build_query_input(Z_per_action[a], X_hat[s], S_inv, XY_mean)
            with torch.no_grad():
                prob = torch.sigmoid(model(torch.from_numpy(inp).unsqueeze(0).to(device))).item()
            Y_i = rng.binomial(1, prob)
            Y_hat_a[s] = Y_i
            stats.update(X_hat[s], Y_i)
        Y_hat_per_action[a] = Y_hat_a
    return Y_hat_per_action


def generate_bandit_task(rng, T, n_actions, d_x=D_X, d_z=D_Z):
    """Formula (10) del paper. Il prezzo non serve: la policy predice solo P(Y=1|X)."""
    X = rng.normal(size=(T, d_x))
    actions = []
    for _ in range(n_actions):
        Z = rng.normal(size=d_z)
        U_const = rng.normal(0.0, 1.0)
        U_Z = rng.normal(1.0, 0.25, size=d_z)
        U_X = rng.normal(1.0, 0.25, size=d_x)
        u_cross = rng.normal(1.0, 0.25, size=2)
        W = U_const + U_Z @ Z + X @ U_X + (X[:, :2] * u_cross * Z[:2]).sum(axis=1)
        Y = rng.binomial(1, sigmoid(W))
        actions.append({"Z": Z, "Y": Y})
    return {"X": X, "actions": actions}


# ----------------------------------------------------------------------------------------------
# FEATURE DEL DATASET IMPUTATO (di UNA azione) -- DEVE essere identica in fase di inferenza
# ----------------------------------------------------------------------------------------------
_IU = np.triu_indices(D_X)
D_FEAT = 2 + D_X + 2 * len(_IU[0]) + D_X     # 42


def dataset_features(X, y, t_ref=T_REF):
    """Riassunto di dimensione fissa del dataset imputato (X: n x 5, y: n).
    [n/t_ref, mean(y), mean(x*y) (5), mean(x x^T) triu (15), mean(y x x^T) triu (15), mean(x) (5)]"""
    n = X.shape[0]
    xy = (X * y[:, None]).mean(axis=0)
    xx = np.einsum("ni,nj->ij", X, X) / n
    xxy = np.einsum("n,ni,nj->ij", y, X, X) / n
    return np.concatenate([[n / t_ref, y.mean()], xy, xx[_IU], xxy[_IU], X.mean(axis=0)]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# INSEGNANTE: stessa procedura di fit_oracle_policies(policy="mlp")
# ----------------------------------------------------------------------------------------------
def teacher_probs(X_hat, y, Q, seed):
    """Fitta l'MLP (default sklearn) su (X_hat, y) e restituisce P(Y=1|q) per ogni riga di Q."""
    if len(np.unique(y)) < 2:
        return np.full(Q.shape[0], float(y.mean())), 1            # caso 'constant'
    clf = MLPClassifier(random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X_hat, y)
    return clf.predict_proba(Q)[:, 1], 0


# ----------------------------------------------------------------------------------------------
# GENERAZIONE PER TASK
# ----------------------------------------------------------------------------------------------
_MODEL = None


def _get_model(path):
    global _MODEL
    if _MODEL is None:
        torch.set_num_threads(1)
        _MODEL = load_p_theta(path)
    return _MODEL


def process_task(task_idx, cfg):
    path = os.path.join(cfg["out_dir"], "tasks", f"task_{task_idx:05d}.npz")
    if os.path.exists(path):
        return path
    t_start = time.time()
    model = _get_model(cfg["p_theta_path"])
    rng = np.random.default_rng(cfg["base_seed"] + task_idx)

    task = generate_bandit_task(rng, T, N_ACTIONS)
    X = task["X"]
    Ymat = np.array([a["Y"] for a in task["actions"]])
    Z = [a["Z"] for a in task["actions"]]

    q_obs = rng.uniform(Q_OBS_MIN, 1.0)                     # densita' della storia
    w = rng.dirichlet(np.ones(N_ACTIONS))                   # sbilanciamento tra azioni
    log_times = set(rng.choice(T, size=TIMES_PER_TASK, replace=False).tolist())

    observed_times = {a: [] for a in range(N_ACTIONS)}
    Y_hist = {a: {} for a in range(N_ACTIONS)}
    F_l, Q_l, P_l, meta_l = [], [], [], []

    for t in range(T):
        if t in log_times:
            eff_T = min(T, t + 1 + HORIZON)
            X_hat = np.zeros((eff_T, D_X))
            X_hat[:t + 1] = X[:t + 1]
            if eff_T > t + 1:
                X_hat[t + 1:] = rng.normal(size=(eff_T - (t + 1), D_X))
            Y_hat = impute_missing_outcomes(model, Z, X_hat, observed_times, Y_hist, N_ACTIONS, rng)

            for a in range(N_ACTIONS):
                y = np.array([Y_hat[a][s] for s in range(eff_T)], dtype=np.float64)
                Q = np.vstack([X[t], rng.normal(size=(K_QUERIES - 1, D_X))])
                P, is_const = teacher_probs(X_hat, y, Q, seed=int(rng.integers(0, 2**31 - 1)))
                F_l.append(dataset_features(X_hat, y))
                Q_l.append(Q.astype(np.float32))
                P_l.append(P.astype(np.float32))
                meta_l.append((task_idx, t, a, eff_T, is_const))

        if rng.random() < q_obs:                            # il "vincitore" osserva
            a = int(rng.choice(N_ACTIONS, p=w))
            observed_times[a].append(t)
            Y_hist[a][t] = int(Ymat[a, t])

    tmp = path.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp, F=np.stack(F_l), Q=np.stack(Q_l), P=np.stack(P_l),
                        meta=np.array(meta_l, dtype=np.int32))
    os.replace(tmp, path)
    print(f"task {task_idx} salvato ({time.time() - t_start:.0f}s, q_obs={q_obs:.2f})", flush=True)
    return path


def merge(out_dir, n_tasks):
    files = [os.path.join(out_dir, "tasks", f"task_{i:05d}.npz") for i in range(n_tasks)]
    files = [f for f in files if os.path.exists(f)]
    parts = [np.load(f) for f in files]
    F = np.concatenate([p["F"] for p in parts])
    Q = np.concatenate([p["Q"] for p in parts])
    P = np.concatenate([p["P"] for p in parts])
    meta = np.concatenate([p["meta"] for p in parts])      # colonne: task, t, azione, n_righe, costante
    out = os.path.join(out_dir, "distill_dataset.npz")
    np.savez_compressed(out, F=F, Q=Q, P=P, meta=meta)
    print(f"\nDataset unito: {out}")
    print(f"  task: {len(files)} | MLP fittati (dataset): {F.shape[0]} | esempi (dataset x contesto): {P.size}")
    print(f"  dimensione feature dataset: {F.shape[1]} | contesto: {Q.shape[2]}")
    print(f"  frazione di dataset 'costanti' (tutti Y uguali): {meta[:, 4].mean():.3f}")
    print(f"  probabilita' insegnante: media {P.mean():.3f}, std {P.std():.3f}, "
          f"quota <0.05 o >0.95: {((P < 0.05) | (P > 0.95)).mean():.3f}")
    return out


# ----------------------------------------------------------------------------------------------
# ESECUZIONE
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, "tasks"), exist_ok=True)
    cfg = {"out_dir": OUT_DIR, "p_theta_path": os.path.abspath(P_THETA_PATH), "base_seed": BASE_SEED}
    t0 = time.time()
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(process_task)(i, cfg) for i in range(N_TASKS))
    print(f"\nGenerazione completata in {(time.time() - t0) / 60:.1f} minuti")
    merge(OUT_DIR, N_TASKS)