"""
Notebook di confronto: asta first-price (pagamento per impression), 5 agenti truthful,
5 slot (nessun agente perde l'asta), catalogo comune di prodotti con prezzo noto.

Il dataset offline e p_theta (p_theta.pt) sono quelli GIA' prodotti: qui si caricano soltanto.
Il file si puo' anche importare (le definizioni sono separate dal blocco "__main__").
"""
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# COSTANTI
# ----------------------------------------------------------------------------------------------
D_X = 5
D_Z = 2
REPEAT_FACTOR = 100
INPUT_DIM = D_Z + D_X + REPEAT_FACTOR * (D_X * D_X + D_X)  # 3007
PRICE_MIN, PRICE_MAX = 6.0, 6.1   # prezzi in unita' monetarie (NON normalizzati)


def sigmoid(w):
    return 1.0 / (1.0 + np.exp(-w))


def to_prob(score, mode="clip"):
    """Trasforma un punteggio lineare in stima di P(Y=1)."""
    score = np.asarray(score, dtype=float)
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-score))
    return np.clip(score, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------
# MODELLO p_theta (deve combaciare con quello allenato)
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


def load_p_theta(path="p_theta.pt", device="cpu"):
    m = SequenceModelMLP(INPUT_DIM).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


class RunningStats:
    """(X^T X + ridge*I)^-1 e MEDIA di X^T Y, aggiornati un'osservazione alla volta
    (stessa logica delle statistiche usate in training)."""
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


# ----------------------------------------------------------------------------------------------
# GENERAZIONE DEL TASK (formula (10) del paper + prezzo casuale NON legato a Y)
# ----------------------------------------------------------------------------------------------
def generate_bandit_task(rng, T, n_actions, d_x=D_X, d_z=D_Z):
    X = rng.normal(size=(T, d_x))
    actions = []
    for _ in range(n_actions):
        Z = rng.normal(size=d_z)
        U_const = rng.normal(0.0, 1.0)
        U_Z = rng.normal(1.0, 0.25, size=d_z)
        U_X = rng.normal(1.0, 0.25, size=d_x)
        u_cross = rng.normal(1.0, 0.25, size=2)

        W = U_const + U_Z @ Z + X @ U_X + (X[:, :2] * u_cross * Z[:2]).sum(axis=1)
        probs = sigmoid(W)
        Y = rng.binomial(1, probs)
        price = rng.uniform(PRICE_MIN, PRICE_MAX)          # unita' monetarie

        actions.append({"Z": Z, "Y": Y, "probs": probs, "price": price})
    return {"X": X, "actions": actions}


# ----------------------------------------------------------------------------------------------
# TS-GEN: imputazione (Algorithm 3) e fitting della policy
# ----------------------------------------------------------------------------------------------
def impute_missing_outcomes(model, Z_per_action, X_hat, observed_times, Y_hist,
                            n_actions, rng, device="cpu"):
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


def fit_oracle_policies(X_hat, Y_hat_per_action, n_actions):
    policies = []
    for a in range(n_actions):
        y = np.array([Y_hat_per_action[a][s] for s in range(X_hat.shape[0])])
        if len(np.unique(y)) < 2:
            policies.append(("constant", float(y.mean())))
        else:
            clf = LogisticRegression()
            clf.fit(X_hat, y)
            policies.append(("model", clf))
    return policies


def predict_probs(policies, X_t):
    return np.array([m if kind == "constant" else m.predict_proba(X_t.reshape(1, -1))[0, 1]
                     for kind, m in policies])


# ----------------------------------------------------------------------------------------------
# AGENTI (con stato). Interfaccia comune:
#   choose(X_t, t) -> (azione, vettore CTR stimati su tutte le azioni)
#   update(a, X_t, Y_t, t)   (feedback bandit: solo per l'azione proposta)
# Strategia di bidding truthful: bid = CTR_stimato(a) * prezzo(a), con a = argmax(CTR*prezzo).
# ----------------------------------------------------------------------------------------------
class BaseAgent:
    name = "base"

    def __init__(self, n_actions, prices, Z_per_action, seed=0):
        self.n_actions = n_actions
        self.prices = np.asarray(prices, dtype=float)
        self.Z = Z_per_action
        self.rng = np.random.default_rng(seed)

    def estimate_ctr(self, X_t, t):
        raise NotImplementedError

    def choose(self, X_t, t):
        ctr = self.estimate_ctr(X_t, t)
        return int(np.argmax(ctr * self.prices)), ctr

    def update(self, a, X_t, Y_t, t):
        raise NotImplementedError


class GreedyAgent(BaseAgent):
    """Usa p_theta sulla sola storia reale (nessuna generazione). epsilon>0 -> Epsilon-Greedy."""
    def __init__(self, model, n_actions, prices, Z_per_action, epsilon=0.0, seed=0, device="cpu"):
        super().__init__(n_actions, prices, Z_per_action, seed)
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.stats = [RunningStats(D_X) for _ in range(n_actions)]
        self.name = "Epsilon-Greedy" if epsilon > 0 else "Greedy"

    def estimate_ctr(self, X_t, t):
        inputs = []
        for a in range(self.n_actions):
            S_inv, XY = self.stats[a].current()
            inputs.append(build_query_input(self.Z[a], X_t, S_inv, XY))
        with torch.no_grad():
            p = torch.sigmoid(self.model(torch.from_numpy(np.stack(inputs)).to(self.device)))
        return p.cpu().numpy().astype(float)

    def choose(self, X_t, t):
        ctr = self.estimate_ctr(X_t, t)
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(0, self.n_actions))
        else:
            a = int(np.argmax(ctr * self.prices))
        return a, ctr

    def update(self, a, X_t, Y_t, t):
        self.stats[a].update(X_t, Y_t)


class TSLinearAgent(BaseAgent):
    """Thompson sampling lineare per azione, prior N(0,I), rumore noise_var."""
    name = "TS-Linear"

    def __init__(self, n_actions, prices, Z_per_action, noise_var=0.25, prob_mode="clip", seed=0):
        super().__init__(n_actions, prices, Z_per_action, seed)
        self.noise_var = noise_var
        self.prob_mode = prob_mode
        self.A = [np.eye(D_X) for _ in range(n_actions)]      # precisione a posteriori
        self.b = [np.zeros(D_X) for _ in range(n_actions)]    # X^T Y / noise_var

    def estimate_ctr(self, X_t, t):
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            Sigma = np.linalg.inv(self.A[a])
            mu = Sigma @ self.b[a]
            beta = self.rng.multivariate_normal(mu, Sigma)
            scores[a] = X_t @ beta
        return to_prob(scores, self.prob_mode)

    def update(self, a, X_t, Y_t, t):
        self.A[a] += np.outer(X_t, X_t) / self.noise_var
        self.b[a] += X_t * Y_t / self.noise_var


class LinUCBAgent(BaseAgent):
    """LinUCB disjoint (Li et al. 2010). Il bid include il bonus ottimistico."""
    name = "LinUCB"

    def __init__(self, n_actions, prices, Z_per_action, alpha=0.1, prob_mode="clip", seed=0):
        super().__init__(n_actions, prices, Z_per_action, seed)
        self.alpha = alpha
        self.prob_mode = prob_mode
        self.A = [np.eye(D_X) for _ in range(n_actions)]
        self.b = [np.zeros(D_X) for _ in range(n_actions)]

    def estimate_ctr(self, X_t, t):
        scores = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = X_t @ theta + self.alpha * np.sqrt(X_t @ A_inv @ X_t)
        return to_prob(scores, self.prob_mode)

    def update(self, a, X_t, Y_t, t):
        self.A[a] += np.outer(X_t, X_t)
        self.b[a] += X_t * Y_t


class TSGenAgent(BaseAgent):
    """Generative TS: a ogni t imputa gli outcome mancanti con p_theta (orizzonte futuro
    troncato a `horizon`), fitta una logistica per azione sul dataset imputato e usa
    le sue probabilita' come CTR stimato."""
    name = "TS-Gen"

    def __init__(self, model, n_actions, prices, Z_per_action, T, horizon, seed=0, device="cpu"):
        super().__init__(n_actions, prices, Z_per_action, seed)
        self.model = model
        self.T = T
        self.horizon = horizon
        self.device = device
        self.X_seen = []                                       # contesti reali visti (indice = t)
        self.observed_times = {a: [] for a in range(n_actions)}
        self.Y_hist = {a: {} for a in range(n_actions)}

    def estimate_ctr(self, X_t, t):
        self.X_seen.append(X_t)
        eff_T = min(self.T, t + 1 + self.horizon)
        X_hat = np.zeros((eff_T, D_X))
        X_hat[:t + 1] = np.array(self.X_seen)
        if eff_T > t + 1:
            X_hat[t + 1:] = self.rng.normal(size=(eff_T - (t + 1), D_X))

        Y_hat = impute_missing_outcomes(self.model, self.Z, X_hat, self.observed_times,
                                        self.Y_hist, self.n_actions, self.rng, self.device)
        policies = fit_oracle_policies(X_hat, Y_hat, self.n_actions)
        return predict_probs(policies, X_t)

    def update(self, a, X_t, Y_t, t):
        self.observed_times[a].append(t)
        self.Y_hist[a][t] = Y_t


# ----------------------------------------------------------------------------------------------
# ASTA FIRST-PRICE (per impression). 5 agenti, 5 slot: tutti vincono sempre.
# Y_t^(a) e' unico per prodotto: se due agenti propongono lo stesso prodotto, l'acquisto
# e' attribuito a entrambi.
# ----------------------------------------------------------------------------------------------
METRICS = ["clicks", "spend", "revenue", "utility", "bias", "regret"]


def run_auction(task, agents, T):
    n_actions = len(task["actions"])
    prices = np.array([a["price"] for a in task["actions"]])
    Ymat = np.array([a["Y"] for a in task["actions"]])           # (n_actions, T)
    Pmat = np.array([a["probs"] for a in task["actions"]])       # (n_actions, T)
    X = task["X"]

    # Oracolo (riferimento esterno): conosce tutti gli outcome, prende il massimo ricavo per t
    oracle_rev = (Ymat * prices[:, None]).max(axis=0)

    out = {ag.name: {m: np.zeros(T) for m in METRICS} for ag in agents}
    out["_oracle_revenue"] = oracle_rev

    for t in range(T):
        X_t = X[t]
        for ag in agents:
            a, ctr = ag.choose(X_t, t)
            bid = ctr[a] * prices[a]                  # truthful: bid = guadagno atteso stimato
            Y = int(Ymat[a, t])
            revenue = Y * prices[a]
            true_value = Pmat[a, t] * prices[a]       # guadagno atteso vero (solo per metrica bias)

            r = out[ag.name]
            r["clicks"][t] = Y
            r["spend"][t] = bid                       # first-price per impression: si paga sempre
            r["revenue"][t] = revenue
            r["utility"][t] = revenue - bid
            r["bias"][t] = bid - true_value
            r["regret"][t] = oracle_rev[t] - revenue
            ag.update(a, X_t, Y, t)                   # perdere il click => Y=0
    return out


def make_agents(model, task, T, horizon, seed, device="cpu"):
    prices = [a["price"] for a in task["actions"]]
    Z = [a["Z"] for a in task["actions"]]
    n = len(prices)
    return [
        TSGenAgent(model, n, prices, Z, T=T, horizon=horizon, seed=seed + 1, device=device),
        GreedyAgent(model, n, prices, Z, epsilon=0.0, seed=seed + 2, device=device),
        GreedyAgent(model, n, prices, Z, epsilon=0.1, seed=seed + 3, device=device),
        TSLinearAgent(n, prices, Z, seed=seed + 4),
        LinUCBAgent(n, prices, Z, seed=seed + 5),
    ]


def run_monte_carlo_auctions(model, n_tasks, T, n_actions, horizon, base_seed=0, device="cpu"):
    results = None
    for m in range(n_tasks):
        t0 = time.time()
        rng_task = np.random.default_rng(base_seed + m)
        task = generate_bandit_task(rng_task, T=T, n_actions=n_actions)   # stesso task per tutti
        agents = make_agents(model, task, T, horizon, seed=1000 * (base_seed + m), device=device)
        out = run_auction(task, agents, T)

        if results is None:
            results = {ag.name: {k: np.zeros((n_tasks, T)) for k in METRICS} for ag in agents}
        for ag in agents:
            for k in METRICS:
                results[ag.name][k][m] = out[ag.name][k]
        print(f"Task {m + 1}/{n_tasks} completato ({time.time() - t0:.0f}s)")
    return results


def summarize(results):
    """Medie (e std) tra task dei totali finali. Il bias e' la media per asta."""
    print(f"\n{'Agente':<16}{'Click':>16}{'Spesa':>18}{'Guadagno lordo':>20}{'Utilita netta':>20}"
          f"{'Bias bid':>16}{'Regret':>18}")
    for name, r in results.items():
        tot = {k: r[k].sum(axis=1) for k in METRICS}
        tot["bias"] = r["bias"].mean(axis=1)
        cells = [f"{tot[k].mean():>9.2f}±{tot[k].std():<6.2f}" for k in METRICS]
        print(f"{name:<16}" + "".join(f"{c:>{w}}" for c, w in zip(cells, [16, 18, 20, 20, 16, 18])))


def plot_results(results, T, n_tasks, path="auction_comparison.png"):
    colors = {"TS-Gen": "tab:blue", "Greedy": "tab:orange", "Epsilon-Greedy": "tab:green",
              "TS-Linear": "tab:red", "LinUCB": "tab:purple"}
    titles = {"clicks": "Click cumulati", "spend": "Spesa cumulata", "revenue": "Guadagno lordo cumulato",
              "utility": "Utilita' netta cumulata", "bias": "Bias medio del bid (running)",
              "regret": "Regret vs oracolo (ricavo)"}
    t_axis = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, k in zip(axes.ravel(), METRICS):
        for name, r in results.items():
            mean_curve = r[k].mean(axis=0)
            y = np.cumsum(mean_curve) / t_axis if k == "bias" else np.cumsum(mean_curve)
            ax.plot(t_axis, y, label=name, color=colors.get(name))
        if k in ("utility", "bias"):
            ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(titles[k])
        ax.set_xlabel("Decision times (t)")
    axes[0, 0].legend()
    fig.suptitle(f"Asta first-price, 5 agenti truthful ({n_tasks} task, T={T})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


# ----------------------------------------------------------------------------------------------
# ESECUZIONE
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"
    model = load_p_theta("p_theta.pt", device)
    print("p_theta caricato, input_dim =", INPUT_DIM)

    # Prima prova veloce: T=30, HORIZON=10, N_TASKS=2. Poi scala a T=300, HORIZON=80, N_TASKS=40.
    T_RUN = 300
    HORIZON = 80
    N_ACTIONS_RUN = 4        # prodotti nel catalogo
    N_TASKS_MC = 30

    results = run_monte_carlo_auctions(model, N_TASKS_MC, T_RUN, N_ACTIONS_RUN, HORIZON,
                                       base_seed=0, device=device)
    summarize(results)
    plot_results(results, T_RUN, N_TASKS_MC)