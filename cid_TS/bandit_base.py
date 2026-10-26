import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. AMBIENTE MULTI-ARMED BANDIT (Gaussiani)
# ==========================================
class GaussianBandit:
    def __init__(self, means, stds):
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.n_arms = len(means)
        self.best_arm = np.argmax(self.means)
        self.best_mean = self.means[self.best_arm]
        
    def pull(self, arm):
        # Restituisce una reward continua da una Gaussiana N(mu, sigma^2)
        return np.random.normal(self.means[arm], self.stds[arm])

# ==========================================
# 2. ALGORITMO C.I.D. THOMPSON SAMPLING
# ==========================================
class CIDThompsonSampling:
    def __init__(self, n_arms, M_steps=100):
        self.n_arms = n_arms
        self.M_steps = M_steps # Orizzonte di simulazione della catena c.i.d.
        self.counts = np.zeros(n_arms, dtype=int)
        self.rewards_history = [[] for _ in range(n_arms)]
        
    def select_arm(self):
        sampled_means = np.zeros(self.n_arms)
        
        for k in range(self.n_arms):
            n_k = self.counts[k]
            
            if n_k == 0:
                # Prior pura (nessuna osservazione): partiamo da N(0,1)
                # Simuliamo una breve catena c.i.d. con alpha_m = 1/m
                x_curr = np.random.normal(0, 1)
                chain = [x_curr]
                for m in range(1, self.M_steps):
                    alpha_m = 1.0 / (m + 1)
                    # Semplificazione del modello c.i.d. normale (Sez. 4.2 del paper)
                    z = np.random.normal(0, 1)
                    x_next = (1 - alpha_m) * chain[-1] + alpha_m * x_curr + np.sqrt(1 - alpha_m**2) * z
                    chain.append(x_next)
                    x_curr = x_next
                sampled_means[k] = np.mean(chain)
                
            else:
                # Posterior: partiamo ancorati ai dati reali osservati sul braccio k
                data = self.rewards_history[k]
                mu_emp = np.mean(data)
                sigma_emp = np.std(data) if np.std(data) > 0 else 1.0
                
                # L'indice m parte da n_k (condizionato ai dati visti)[cite: 13]
                x_curr = np.random.normal(mu_emp, sigma_emp)
                chain = [x_curr]
                for step in range(self.M_steps):
                    m = n_k + step
                    alpha_m = 1.0 / m  # Il peso decresce man mano che m è grande
                    z = np.random.normal(0, 1)
                    x_next = (1 - alpha_m) * chain[-1] + alpha_m * mu_emp + np.sqrt(max(0, 1 - alpha_m**2)) * z * (sigma_emp / np.sqrt(m))
                    chain.append(x_next)
                    x_curr = x_next
                
                # Il campione della posterior è la media della traiettoria limite c.i.d.
                sampled_means[k] = np.mean(chain)
                
        return np.argmax(sampled_means)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards_history[arm].append(reward)

# ==========================================
# 3. ALGORITMO STANDARD GAUSSIAN TS (Baseline)
# ==========================================
class ConjugateGaussianTS:
    def __init__(self, n_arms, prior_mu=0.0, prior_var=1.0, likelihood_var=1.0):
        self.n_arms = n_arms
        self.mu_0 = prior_mu
        self.sigma0_sq = prior_var
        self.sigma_sq = likelihood_var
        
        self.n = np.zeros(n_arms)
        self.sum_x = np.zeros(n_arms)
        
    def select_arm(self):
        sampled_means = np.zeros(self.n_arms)
        for k in range(self.n_arms):
            n_k = self.n[k]
            if n_k == 0:
                post_mu = self.mu_0
                post_var = self.sigma0_sq
            else:
                mean_k = self.sum_x[k] / n_k
                # Formula esatta della posterior gaussiana coniugata
                post_var = 1.0 / ( (1.0 / self.sigma0_sq) + (n_k / self.sigma_sq) )
                post_mu = post_var * ( (self.mu_0 / self.sigma0_sq) + (n_k * mean_k / self.sigma_sq) )
            
            sampled_means[k] = np.random.normal(post_mu, np.sqrt(post_var))
        return np.argmax(sampled_means)
    
    def update(self, arm, reward):
        self.n[arm] += 1
        self.sum_x[arm] += reward

# ==========================================
# 4. ALGORITMO UCB1 (Baseline deterministica)
# ==========================================
class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        
    def select_arm(self):
        self.t += 1
        # Esplora inizialmente ogni braccio almeno una volta
        for k in range(self.n_arms):
            if self.counts[k] == 0:
                return k
                
        ucb_values = self.values + np.sqrt(2 * np.log(self.t) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = value + (reward - value) / n

# ==========================================
# 5. ESECUZIONE DELLA SIMULAZIONE E CONFRONTO
# ==========================================
def run_simulation(T=2000, n_simulations=20):
    true_means = [1.2, 1.8, 1.5] # Medie vere dei 3 bracci
    true_stds = [1.0, 1.0, 1.0]  # Varianze note
    
    regrets = {
        "CID-Thompson Sampling": np.zeros(T),
        "Conjugate Gaussian TS": np.zeros(T),
        "UCB1": np.zeros(T)
    }
    
    for sim in range(n_simulations):
        print(f"Simulazione {sim+1}/{n_simulations}...")
        
        bandit = GaussianBandit(true_means, true_stds)
        agents = {
            "CID-Thompson Sampling": CIDThompsonSampling(bandit.n_arms, M_steps=500),
            "Conjugate Gaussian TS": ConjugateGaussianTS(bandit.n_arms),
            "UCB1": UCB1(bandit.n_arms)
        }
        
        for name, agent in agents.items():
            cumulative_regret = 0
            regret_path = []
            
            for t in range(1, T + 1):
                chosen_arm = agent.select_arm()
                reward = bandit.pull(chosen_arm)
                agent.update(chosen_arm, reward)
                
                # Calcolo del regret istantaneo: (media ottima - media del braccio scelto)
                instant_regret = bandit.best_mean - bandit.means[chosen_arm]
                cumulative_regret += instant_regret
                regret_path.append(cumulative_regret)
                
            regrets[name] += np.array(regret_path)
            
    # Media sui cicli di simulazione
    for name in regrets:
        regrets[name] /= n_simulations
        
    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    for name, regret_path in regrets.items():
        plt.plot(range(1, T + 1), regret_path, label=name, linewidth=2)
        
    plt.xlabel("Tempo (Step T)", fontsize=12)
    plt.ylabel("Cumulative Regret Medio", fontsize=12)
    plt.title("Confronto Cumulative Regret: c.i.d.-TS vs Baseline", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

# Avvia l'esperimento (puoi variare T o il numero di simulazioni)
if __name__ == "__main__":
    run_simulation(T=100, n_simulations=10)