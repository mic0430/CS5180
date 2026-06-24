import numpy as np
import gymnasium as gym


def policy_evaluation_linear(pi, P, gamma):
    """Solve V^pi = (I - gamma P_pi)^-1 R_pi via numpy.linalg.solve."""
    nS = len(P)
    P_pi = np.zeros((nS, nS))
    R_pi = np.zeros(nS)
    for s in range(nS):
        a = pi[s]
        for prob, next_s, reward, terminated in P[s][a]:
            R_pi[s] += prob * reward
            if not terminated:
                P_pi[s, next_s] += prob
    A = np.eye(nS) - gamma * P_pi
    return np.linalg.solve(A, R_pi)


def policy_improvement(V, P, gamma):
    """Greedy improvement: pi'(s) = argmax_a sum_{s'} P[s,a,s'](R + gamma V[s'])."""
    nS = len(P)
    nA = len(P[0])
    pi_new = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_s, reward, terminated in P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_s] * (not terminated))
        pi_new[s] = int(np.argmax(q_values))
    return pi_new


def policy_iteration(P, gamma):
    """Standard PI loop: evaluate, improve, terminate when policy unchanged."""
    nS = len(P)
    pi = np.zeros(nS, dtype=int)  # initial: all-Left
    iter_count = 0
    while True:
        iter_count += 1
        V = policy_evaluation_linear(pi, P, gamma)
        pi_new = policy_improvement(V, P, gamma)
        if np.array_equal(pi_new, pi):
            return V, pi, iter_count
        pi = pi_new


def render_policy(pi, shape=(4, 4)):
    arrows = {0: "<", 1: "v", 2: ">", 3: "^"}
    grid = pi.reshape(shape)
    return "\n".join("  ".join(arrows[a] for a in grid[r]) for r in range(shape[0]))


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99
    V_star, pi_star, iters = policy_iteration(P, gamma)

    print(f"Policy Iteration on FrozenLake-v1 (gamma={gamma})")
    print(f"Iterations to converge: {iters}")
    print(f"\nV* table (4x4 grid):")
    print(V_star.reshape(4, 4).round(4))
    print(f"\npi* as arrows:")
    print(render_policy(pi_star))
    print(f"\npi* as integer array: {pi_star}")