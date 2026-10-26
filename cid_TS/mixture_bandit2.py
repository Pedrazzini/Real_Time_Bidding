import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# ==========================================
# 1. AMBIENTE BANDIT CON MISCLELE
# ==========================================
class MixtureBandit:
    def __init__(self):
        self.n_arms = 2
        
    def pull(self, arm):
        if arm == 0:
            # Braccio 0 (Ottimale): Miscela 50% N(-1.5, 0.5^2) e 50% N(2.5, 0.5^2)
            if np.random.rand() < 0.5:
                return np.random.normal(-1.5, 0.5)
            else:
                return np.random.normal(2.5, 0.5)
        else:
            # Braccio 1 (Sub-ottimale): Gaussiana semplice N(0.5, 1.0^2)
            return np.random.normal(0.5, 1.0)

# ==========================================
# 2. CID-TS PER MISCELE (Tempo O(1) ricorsivo)
# ==========================================
class MixtureCIDThompsonSampling:
    def __init__(self, n_arms, n_components=2):
        self.n_arms = n_arms
        self.n_comp = n_components
        # Inizializzazione ricorsiva c.i.d. (Sezione 4.2 del paper)
        self.weights = [np.ones(n_components) / n_components for _ in range(n_arms)]
        self.thetas = [np.array([-1.0, 1.0]) for _ in range(n_arms)] 
        self.sigma_sq = 0.5
        self.counts = np.zeros(n_arms, dtype=int)
        
    def select_arm(self):
        sampled_means = np.zeros(self.n_arms)
        for k in range(self.n_arms):
            chosen_comp = np.random.choice(self.n_comp, p=self.weights[k])
            sampled_means[k] = np.random.normal(self.thetas[k][chosen_comp], np.sqrt(self.sigma_sq))
        return np.argmax(sampled_means)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        m = self.counts[arm]
        a_m = 1.0 / m  # Peso ricorsivo decrescente[cite: 16, 17]
        
        w = self.weights[arm]
        thetas = self.thetas[arm]
        
        # Aggiornamento in tempo costante O(1) senza ri-processare lo storico[cite: 16, 17]
        densities = np.array([np.exp(-0.5 * (reward - th)**2 / self.sigma_sq) for th in thetas])
        new_w = w * densities
        if np.sum(new_w) > 0:
            new_w /= np.sum(new_w)
        else:
            new_w = w
        self.weights[arm] = new_w
        
        best_comp = np.argmax(new_w)
        self.thetas[arm][best_comp] = (1 - a_m) * self.thetas[arm][best_comp] + a_m * reward

# ==========================================
# 3. BASELINE: GMM-TS con Expectation-Maximization (Pesante Computazionalmente)
# ==========================================
class GMMThompsonSampling:
    def __init__(self, n_arms, n_components=2):
        self.n_arms = n_arms
        self.n_comp = n_components
        self.history = [[] for _ in range(n_arms)]
        
    def select_arm(self):
        sampled_means = np.zeros(self.n_arms)
        for k in range(self.n_arms):
            data = self.history[k]
            # Se ci sono pochi dati (meno di 4), usa un'estrazione casuale iniziale
            if len(data) < 4:
                sampled_means[k] = np.random.normal(0, 1)
            else:
                # Esegue il fitting dell'algoritmo EM sull'INTERO storico dati raccolto finora
                X = np.array(data).reshape(-1, 1)
                gmm = GaussianMixture(n_components=self.n_comp, random_state=42, max_iter=20)
                try:
                    gmm.fit(X)
                    # Campiona un componente e genera una reward prevista
                    chosen_comp = np.random.choice(self.n_comp, p=gmm.weights_)
                    mu = gmm.means_[chosen_comp][0]
                    var = gmm.covariances_[chosen_comp][0][0]
                    sampled_means[k] = np.random.normal(mu, np.sqrt(max(1e-3, var)))
                except:
                    sampled_means[k] = np.mean(data)
        return np.argmax(sampled_means)
    
    def update(self, arm, reward):
        # Accumula i dati: a ogni step lo storico cresce, aumentando il costo di gmm.fit()
        self.history[arm].append(reward)

# ==========================================
# 4. SIMULAZIONE E CONFRONTO DELLE PRESTAZIONI
# ==========================================
def run_benchmark(T=500, n_simulations=5):
    print("=== INIZIO BENCHMARK REGRET E TEMPI COMPUTAZIONALI ===")
    
    regrets = {
        "CID-TS (Ricorsivo O(1))": np.zeros(T),
        "GMM-TS (EM Re-fitting)": np.zeros(T)
    }
    
    execution_times = {
        "CID-TS (Ricorsivo O(1))": 0.0,
        "GMM-TS (EM Re-fitting)": 0.0
    }
    
    for sim in range(n_simulations):
        print(f"\n[Simulazione {sim+1}/{n_simulations}]")
        
        bandit = MixtureBandit()
        agents = {
            "CID-TS (Ricorsivo O(1))": MixtureCIDThompsonSampling(bandit.n_arms),
            "GMM-TS (EM Re-fitting)": GMMThompsonSampling(bandit.n_arms)
        }
        
        for name, agent in agents.items():
            start_time = time.time()
            
            cumulative_regret = 0
            regret_path = []
            
            for t in range(1, T + 1):
                chosen_arm = agent.select_arm()
                reward = bandit.pull(chosen_arm)
                agent.update(chosen_arm, reward)
                
                # Regret istantaneo rispetto al braccio 0 (migliore in media sui picchi)
                best_arm_true_mean = 1.0
                chosen_arm_true_mean = 1.0 if chosen_arm == 0 else 0.5
                instant_regret = max(0, best_arm_true_mean - chosen_arm_true_mean)
                
                cumulative_regret += instant_regret
                regret_path.append(cumulative_regret)
                
            elapsed = time.time() - start_time
            execution_times[name] += elapsed
            regrets[name] += np.array(regret_path)
            
            print(f"  -> {name}: completato in {elapsed:.4f} secondi")
            
    # Calcolo medie sui cicli
    for name in regrets:
        regrets[name] /= n_simulations
        execution_times[name] /= n_simulations
        
    print("\n" + "="*50)
    print(" RISULTATI FINALI SUI TEMPI DI ESECUZIONE MEDI:")
    print("="*50)
    for name, t_exec in execution_times.items():
        print(f" * {name:25s}: {t_exec:.4f} secondi per episodio")
    print("="*50)
    
    # Plotting dei Regret
    plt.figure(figsize=(10, 6))
    for name, regret_path in regrets.items():
        plt.plot(range(1, T + 1), regret_path, label=f"{name} ({execution_times[name]:.2f}s)", linewidth=2)
        
    plt.xlabel("Tempo (Step T)", fontsize=12)
    plt.ylabel("Cumulative Regret Medio", fontsize=12)
    plt.title("Confronto Regret e Tempi: CID-TS (Martingala) vs GMM-TS (EM)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

if __name__ == "__main__":
    # T=500 per apprezzare il divario di tempo senza attendere troppo
    run_benchmark(T=500, n_simulations=5)