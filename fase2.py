import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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
    Un task bandit completo VERO: contesti X_1:T condivisi da tutte le azioni,
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


############################################################################################################
#COMPARISON#
############################################################################################################

#GREEDY AND EPSILON-GREEDY
class ActionHistory:
    """Tiene traccia delle sole osservazioni REALI per una singola azione."""
    def __init__(self, d_x):
        self.d_x = d_x
        self.X_list = []
        self.Y_list = []

    def add(self, X_t, Y_t):
        self.X_list.append(X_t)
        self.Y_list.append(Y_t)

    def as_arrays(self):
        if len(self.X_list) == 0:
            return np.zeros((0, self.d_x)), np.zeros((0,))
        return np.array(self.X_list), np.array(self.Y_list)

def predict_with_p_theta(model, Z_a, X_t, hist_a, device="cpu", ridge=1.0):
    """Predice E[Y|storia reale, X_t] con p_theta, SENZA generare nulla (query singola)."""
    X_obs, Y_obs = hist_a.as_arrays()
    stats = RunningStats(hist_a.d_x, ridge=ridge)
    for X_i, Y_i in zip(X_obs, Y_obs):
        stats.update(X_i, Y_i)
    S_inv, XY_mean = stats.current()
    input_vec = build_query_input(Z_a, X_t, S_inv, XY_mean)
    with torch.no_grad():
        logit = model(torch.from_numpy(input_vec).unsqueeze(0).to(device))
        return torch.sigmoid(logit).item()


def run_greedy(model, task, T, n_actions, epsilon=0.0, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    X_true = task["X"]
    Z_per_action = [a["Z"] for a in task["actions"]]
    histories = [ActionHistory(D_X) for _ in range(n_actions)]
    rewards, actions = [], []

    for t in range(T):
        X_t = X_true[t]
        if epsilon > 0 and rng.random() < epsilon:
            A_t = int(rng.integers(0, n_actions))
        else:
            preds = [predict_with_p_theta(model, Z_per_action[a], X_t, histories[a], device)
                     for a in range(n_actions)]
            A_t = int(np.argmax(preds))

        Y_t = int(task["actions"][A_t]["Y"][t])
        histories[A_t].add(X_t, Y_t)
        rewards.append(Y_t)
        actions.append(A_t)

    return {"rewards": rewards, "actions": actions}


#TS-LINEAR
def run_ts_linear(task, T, n_actions, d_x=D_X, noise_var=0.25, seed=0):
    """
    Bayesian linear regression per azione, prior N(0, I), rumore N(0, noise_var).
    Non usa p_theta: lavora solo su (X_t) e sugli outcome realmente osservati.
    """
    rng = np.random.default_rng(seed)
    X_true = task["X"]
    histories = [ActionHistory(d_x) for _ in range(n_actions)]
    rewards, actions = [], []

    for t in range(T):
        X_t = X_true[t]
        preds = []
        for a in range(n_actions):
            X_obs, Y_obs = histories[a].as_arrays()
            if X_obs.shape[0] == 0:
                Sigma = np.eye(d_x)
                mu = np.zeros(d_x)
            else:
                precision = np.eye(d_x) + (X_obs.T @ X_obs) / noise_var
                Sigma = np.linalg.inv(precision)
                mu = Sigma @ (X_obs.T @ Y_obs) / noise_var
            beta_sample = rng.multivariate_normal(mu, Sigma)
            preds.append(X_t @ beta_sample)
        A_t = int(np.argmax(preds))

        Y_t = int(task["actions"][A_t]["Y"][t])
        histories[A_t].add(X_t, Y_t)
        rewards.append(Y_t)
        actions.append(A_t)

    return {"rewards": rewards, "actions": actions}


#LIN-UCB
def run_linucb(task, T, n_actions, d_x=D_X, alpha=0.1, seed=0):
    X_true = task["X"]
    histories = [ActionHistory(d_x) for _ in range(n_actions)]
    rewards, actions = [], []

    for t in range(T):
        X_t = X_true[t]
        ucb_scores = []
        for a in range(n_actions):
            X_obs, Y_obs = histories[a].as_arrays()
            A_mat = np.eye(d_x) + (X_obs.T @ X_obs if X_obs.shape[0] > 0 else 0)
            b_vec = X_obs.T @ Y_obs if X_obs.shape[0] > 0 else np.zeros(d_x)
            A_inv = np.linalg.inv(A_mat)
            theta_hat = A_inv @ b_vec
            mean_est = X_t @ theta_hat
            bonus = alpha * np.sqrt(X_t @ A_inv @ X_t)
            ucb_scores.append(mean_est + bonus)
        A_t = int(np.argmax(ucb_scores))

        Y_t = int(task["actions"][A_t]["Y"][t])
        histories[A_t].add(X_t, Y_t)
        rewards.append(Y_t)
        actions.append(A_t)

    return {"rewards": rewards, "actions": actions}

#RUN ALL

ALGORITHMS = {
    "TS-Gen": lambda model, task, T, n_actions, seed, device:
        generative_ts(model, task, T=T, n_actions=n_actions, horizon=HORIZON,
                      seed=seed, device=device, verbose=False),
    "Greedy": lambda model, task, T, n_actions, seed, device:
        run_greedy(model, task, T, n_actions, epsilon=0.0, seed=seed, device=device),
    "Epsilon-Greedy": lambda model, task, T, n_actions, seed, device:
        run_greedy(model, task, T, n_actions, epsilon=0.1, seed=seed, device=device),
    "TS-Linear": lambda model, task, T, n_actions, seed, device:
        run_ts_linear(task, T, n_actions, seed=seed),
    "LinUCB": lambda model, task, T, n_actions, seed, device:
        run_linucb(task, T, n_actions, seed=seed),
}

def run_monte_carlo_all(model, algo_names, n_tasks, T, n_actions, base_seed=0, device="cpu"):
    results = {name: np.zeros((n_tasks, T)) for name in algo_names}

    for m in range(n_tasks):
        rng_task = np.random.default_rng(base_seed + m)
        task = generate_bandit_task(rng_task, T=T, n_actions=n_actions)  # STESSO task per tutti

        for name in algo_names:
            algo_fn = ALGORITHMS[name]
            out = algo_fn(model, task, T, n_actions, base_seed + m, device)
            regret_info = compute_regret(task, out, n_actions)
            results[name][m] = regret_info["cumulative_regret"]

        print(f"Task {m+1}/{n_tasks} completato")

    mean_results = {name: results[name].mean(axis=0) for name in algo_names}
    std_results = {name: results[name].std(axis=0) for name in algo_names}
    return results, mean_results, std_results

T_RUN = 300
HORIZON = 80
N_ACTIONS_RUN = 4
N_TASKS_MC = 40

algo_names = ["TS-Gen", "Greedy", "Epsilon-Greedy", "TS-Linear", "LinUCB"]

all_results, mean_results, std_results = run_monte_carlo_all(
    model, algo_names, n_tasks=N_TASKS_MC, T=T_RUN, n_actions=N_ACTIONS_RUN,
    base_seed=0, device=device
)

for name in algo_names:
    print(f"{name}: regret finale = {mean_results[name][-1]:.2f} ± {std_results[name][-1]:.2f}")

#PLOT
t_axis = np.arange(1, T_RUN + 1)
colors = {"TS-Gen": "tab:blue", "Greedy": "tab:orange", "Epsilon-Greedy": "tab:green",
          "TS-Linear": "tab:red", "LinUCB": "tab:purple"}

plt.figure(figsize=(8, 6))
for name in algo_names:
    plt.plot(t_axis, mean_results[name], label=name, color=colors[name])
    

plt.xlabel("Decision times (t)")
plt.ylabel("Regret")
plt.title(f"Average Regret Over Timesteps ({N_TASKS_MC} task, T={T_RUN})")
plt.legend()
plt.tight_layout()
plt.savefig("regret_comparison.png", dpi=150)
plt.show()