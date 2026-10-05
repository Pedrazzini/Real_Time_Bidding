import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

def sigmoid(w):
    return 1.0 / (1.0 + np.exp(-w))

class SequenceModelMLP(nn.Module):
    """Stessa architettura usata in training: deve combaciare per caricare i pesi salvati."""
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

D_X = 5
D_Z = 2
REPEAT_FACTOR = 100
INPUT_DIM = D_Z + D_X + REPEAT_FACTOR * (D_X * D_X + D_X)  # 2+5+100*30 = 3007

device = "cpu"
model = SequenceModelMLP(INPUT_DIM).to(device)
model.load_state_dict(torch.load("p_theta.pt", map_location=device))
model.eval()
print("p_theta caricato, input_dim =", INPUT_DIM)

class RunningStats:
    """
    Mantiene (X^T X + I)^-1 e la MEDIA di X^T Y, aggiornate un'osservazione alla volta.
    Deve essere identica, in logica, a build_summary_stats usata in training
    (versione con normalizzazione a media, non somma grezza).
    """
    def __init__(self, d_x, ridge=1.0):
        self.d_x = d_x
        self.ridge = ridge
        self.sum_outer = np.zeros((d_x, d_x))
        self.sum_XY = np.zeros(d_x)
        self.count = 0

    def current(self):
        S = self.sum_outer + self.ridge * np.eye(self.d_x)
        S_inv = np.linalg.inv(S)
        XY_mean = self.sum_XY / max(self.count, 1)
        return S_inv, XY_mean

    def update(self, X_i, Y_i):
        self.sum_outer += np.outer(X_i, X_i)
        self.sum_XY += X_i * Y_i
        self.count += 1


def build_query_input(Z, X_i, S_inv, XY_mean, repeat_factor=REPEAT_FACTOR):
    """Stesso formato di input usato in training: [Z, X_t, summary ripetuta]."""
    summary = np.concatenate([S_inv.flatten(), XY_mean])
    summary_repeated = np.tile(summary, repeat_factor)
    return np.concatenate([Z, X_i, summary_repeated]).astype(np.float32)

def generate_bandit_task(rng, T, n_actions, d_x=D_X, d_z=D_Z):
    """
    Un task bandit completo: contesti X_1:T condivisi da tutte le azioni,
    e per ogni azione l'intera tabella di potential outcomes Y_1:T^(a)
    (formula (10), Appendix B.1.1). L'agente vedrà solo un sottoinsieme
    di questi outcome, in base a quali azioni sceglie nel tempo.
    """
    X = rng.normal(size=(T, d_x))
    actions = []
    for a in range(n_actions):
        Z = rng.normal(size=d_z)
        U_const = rng.normal(0.0, 1.0)
        U_Z = rng.normal(1.0, 0.25, size=d_z)
        U_X = rng.normal(1.0, 0.25, size=d_x)
        u_cross = rng.normal(1.0, 0.25, size=2)

        W = (U_const + U_Z @ Z + X @ U_X
             + (X[:, :2] * u_cross * Z[:2]).sum(axis=1))
        probs = sigmoid(W)
        Y = rng.binomial(1, probs)
        actions.append({"Z": Z, "Y": Y, "probs": probs})

    return {"X": X, "actions": actions}

def impute_missing_outcomes(model, Z_per_action, X_hat, observed_times, Y_hist,
                            n_actions, rng, device="cpu"):
    """
    Algorithm 3. X_hat contiene già sia i contesti reali (osservati) sia quelli
    futuri campionati; questa funzione riempie gli outcome mancanti azione per azione,
    rispettando l'ordinamento: osservati prima, poi mancanti in ordine temporale.
    """
    effective_T = X_hat.shape[0]
    Y_hat_per_action = {}

    for a in range(n_actions):
        obs = sorted(observed_times[a])
        missing = [s for s in range(effective_T) if s not in observed_times[a]]

        stats = RunningStats(D_X)
        for s in obs:
            stats.update(X_hat[s], Y_hist[a][s])

        Y_hat_a = dict(Y_hist[a])  # parte con gli outcome realmente osservati
        for s in missing:
            S_inv, XY_mean = stats.current()
            input_vec = build_query_input(Z_per_action[a], X_hat[s], S_inv, XY_mean)
            with torch.no_grad():
                logit = model(torch.from_numpy(input_vec).unsqueeze(0).to(device))
                prob = torch.sigmoid(logit).item()
            Y_i = rng.binomial(1, prob)
            Y_hat_a[s] = Y_i
            stats.update(X_hat[s], Y_i)

        Y_hat_per_action[a] = Y_hat_a

    return Y_hat_per_action

def fit_oracle_policies(X_hat, Y_hat_per_action, n_actions):
    policies = []
    for a in range(n_actions):
        y = np.array([Y_hat_per_action[a][s] for s in range(X_hat.shape[0])])
        if len(np.unique(y)) < 2:
            policies.append(("constant", float(y.mean())))  # caso degenere: tutti 0 o tutti 1
        else:
            clf = LogisticRegression()
            clf.fit(X_hat, y)
            policies.append(("model", clf))
    return policies

def predict_probs(policies, X_t):
    probs = []
    for kind, m in policies:
        probs.append(m if kind == "constant" else m.predict_proba(X_t.reshape(1, -1))[0, 1])
    return np.array(probs)

def generative_ts(model, task, T, n_actions, horizon, seed=0, device="cpu", verbose=True):
    rng = np.random.default_rng(seed)
    X_true = task["X"]
    Z_per_action = [a["Z"] for a in task["actions"]]

    observed_times = {a: [] for a in range(n_actions)}
    Y_hist = {a: {} for a in range(n_actions)}
    rewards, chosen_actions = [], []

    for t in range(T):
        X_t = X_true[t]
        effective_T = min(T, t + 1 + horizon)  # passato completo + finestra futura troncata

        X_hat = np.zeros((effective_T, D_X))
        X_hat[:t + 1] = X_true[:t + 1]
        if effective_T > t + 1:
            X_hat[t + 1:] = rng.normal(size=(effective_T - (t + 1), D_X))

        Y_hat_per_action = impute_missing_outcomes(
            model, Z_per_action, X_hat, observed_times, Y_hist, n_actions, rng, device
        )

        policies = fit_oracle_policies(X_hat, Y_hat_per_action, n_actions)
        probs_pred = predict_probs(policies, X_t)
        A_t = int(np.argmax(probs_pred))

        Y_t = int(task["actions"][A_t]["Y"][t])
        observed_times[A_t].append(t)
        Y_hist[A_t][t] = Y_t

        rewards.append(Y_t)
        chosen_actions.append(A_t)

        if verbose:
            print(f"t={t:2d} | azione scelta={A_t} | probs stimate={np.round(probs_pred,3)} | Y osservato={Y_t}")

    return {"rewards": rewards, "actions": chosen_actions}

#rng_task = np.random.default_rng(42)
#T_TEST = 100
#N_ACTIONS_TEST = 3
#HORIZON = 30  # troncamento: genera al massimo 10 passi nel futuro oltre a t

#task = generate_bandit_task(rng_task, T=T_TEST, n_actions=N_ACTIONS_TEST)

#result = generative_ts(model, task, T=T_TEST, n_actions=N_ACTIONS_TEST,
#                       horizon=HORIZON, seed=0, device=device)

#print("\nReward totale:", sum(result["rewards"]))
#print("Reward medio:", np.mean(result["rewards"]))
#print("Distribuzione azioni scelte:", np.bincount(result["actions"], minlength=N_ACTIONS_TEST))

#######################################################################################################################
#REGRET COMPUTATION#
#######################################################################################################################

def fit_full_oracle_policy(X_true, task, n_actions):
    """
    Fitta la policy "best-in-hindsight" pi*(.;tau) sull'INTERA tabella di
    potential outcomes vera (non imputata), come richiesto dalla definizione
    di regret in (2). Riusa fit_oracle_policies, fingendo che la tabella
    vera sia "il dataset imputato" (qui non c'è nulla da imputare: è completa).
    """
    T = X_true.shape[0]
    Y_true_dict = {
        a: {s: int(task["actions"][a]["Y"][s]) for s in range(T)}
        for a in range(n_actions)
    }
    return fit_oracle_policies(X_true, Y_true_dict, n_actions)


def compute_regret(task, result, n_actions):
    X_true = task["X"]
    T = X_true.shape[0]

    oracle_policies = fit_full_oracle_policy(X_true, task, n_actions)

    oracle_actions, oracle_rewards = [], []
    for t in range(T):
        probs = predict_probs(oracle_policies, X_true[t])
        a_star = int(np.argmax(probs))
        oracle_actions.append(a_star)
        oracle_rewards.append(int(task["actions"][a_star]["Y"][t]))

    agent_rewards = np.array(result["rewards"])
    oracle_rewards = np.array(oracle_rewards)

    per_period_regret = oracle_rewards - agent_rewards      # R(Y^pi*) - R(Y^At), R(y)=y
    cumulative_regret = np.cumsum(per_period_regret)

    return {
        "oracle_actions": oracle_actions,
        "oracle_rewards": oracle_rewards,
        "per_period_regret": per_period_regret,
        "cumulative_regret": cumulative_regret,
        "avg_regret": cumulative_regret / np.arange(1, T + 1),
    }

#regret_info = compute_regret(task, result, N_ACTIONS_TEST)

#print("Azioni oracle:       ", regret_info["oracle_actions"])
#print("Azioni agente (TS):  ", result["actions"])
#print("Regret per periodo:  ", regret_info["per_period_regret"])
#print("Regret cumulativo:   ", regret_info["cumulative_regret"])
#print(f"\nRegret cumulativo finale: {regret_info['cumulative_regret'][-1]}")
#print(f"Regret medio per periodo (finale): {regret_info['avg_regret'][-1]:.3f}")    


##################################################################################################
#PER ESSERE PRECISI FACCIAMO LA MEDIA DEI RISULTATI OTTENUTI SU PIU' TASK DIVERSI#
##################################################################################################

def run_monte_carlo(model, n_tasks, T, n_actions, horizon, base_seed=0, device="cpu"):
    all_cum_regret = np.zeros((n_tasks, T))

    for m in range(n_tasks):
        rng_task = np.random.default_rng(base_seed + m)
        task = generate_bandit_task(rng_task, T=T, n_actions=n_actions)

        result = generative_ts(model, task, T=T, n_actions=n_actions,
                               horizon=horizon, seed=base_seed + m,
                               device=device, verbose=False)

        regret_info = compute_regret(task, result, n_actions)
        all_cum_regret[m] = regret_info["cumulative_regret"]

        print(f"Task {m+1}/{n_tasks} completato - regret cumulativo finale: {all_cum_regret[m, -1]:.0f}")

    mean_regret = all_cum_regret.mean(axis=0)
    std_regret = all_cum_regret.std(axis=0)
    return all_cum_regret, mean_regret, std_regret


N_TASKS_MC = 40   # inizia piccolo, poi scala se i tempi lo permettono
all_regret, mean_regret, std_regret = run_monte_carlo(
    model, n_tasks=N_TASKS_MC, T=300, n_actions=4, horizon=80, device=device
)

print(f"\nRegret cumulativo medio finale: {mean_regret[-1]:.2f} ± {std_regret[-1]:.2f}")

####################################################################################################
#GRAFICO#
####################################################################################################

import matplotlib.pyplot as plt

T_plot = mean_regret.shape[0]
t_axis = np.arange(1, T_plot + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Grafico 1: regret cumulativo medio, con banda di errore ---
axes[0].plot(t_axis, mean_regret, label="Regret cumulativo medio (TS-Gen)", color="tab:blue")
axes[0].fill_between(t_axis, mean_regret - std_regret, mean_regret + std_regret,
                     alpha=0.2, color="tab:blue", label="±1 std")

# Curva di riferimento c*sqrt(t), scalata per passare vicino all'ultimo punto
c = mean_regret[-1] / np.sqrt(T_plot)
axes[0].plot(t_axis, c * np.sqrt(t_axis), "--", color="gray", label=r"riferimento $c\sqrt{t}$")

axes[0].set_xlabel("Decision times (t)")
axes[0].set_ylabel("Regret cumulativo")
axes[0].set_title("Regret cumulativo medio su task")
axes[0].legend()

# --- Grafico 2: regret medio PER PERIODO (deve tendere a scendere verso 0) ---
avg_per_period = mean_regret / t_axis
axes[1].plot(t_axis, avg_per_period, color="tab:orange")
axes[1].set_xlabel("Decision times (t)")
axes[1].set_ylabel("Regret medio per periodo")
axes[1].set_title("Regret / t (deve tendere a calare)")

plt.tight_layout()
plt.savefig("regret_plot.png", dpi=150)
plt.show()
