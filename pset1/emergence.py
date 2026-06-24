import numpy as np
import gymnasium as gym
from vi import value_iteration


def vi_with_history(P, gamma, theta, V_star):
    
    nS, nA = len(P), len(P[0])
    V = np.zeros(nS)
    bound = theta * (1.0 - gamma) / gamma
    history = []

    while True:
        V_new = np.zeros(nS)
        for s in range(nS):
            qs = np.zeros(nA)
            for a in range(nA):
                for prob, ns, r, term in P[s][a]:
                    qs[a] += prob * (r + gamma * V[ns] * (not term))
            V_new[s] = qs.max()

        pi_k = np.zeros(nS, dtype=int)
        for s in range(nS):
            qs = np.zeros(nA)
            for a in range(nA):
                for prob, ns, r, term in P[s][a]:
                    qs[a] += prob * (r + gamma * V_new[ns] * (not term))
            pi_k[s] = int(np.argmax(qs))

        err = np.max(np.abs(V_new - V_star))
        history.append((V_new.copy(), pi_k.copy(), err))
        if np.max(np.abs(V_new - V)) < bound:
            break
        V = V_new
    return history


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    gamma, theta = 0.99, 1e-4

    V_star, pi_star, iters = value_iteration(P, gamma, theta)
    print(f"VI converged at iter {iters}, pi*: {pi_star}\n")

    history = vi_with_history(P, gamma, theta, V_star)
    for k, (_, pi_k, err) in enumerate(history, start=1):
        if np.array_equal(pi_k, pi_star):
            print(f"k* = {k}  (first iteration where pi_k = pi*)")
            print(f"||V_k* - V*||_inf = {err:.6f}")
            print(f"Theoretical policy-loss bound: 2*gamma*err/(1-gamma) = "
                  f"{2*gamma*err/(1-gamma):.4f}")
            break