import numpy as np
import gymnasium as gym


def value_iteration(P, gamma, theta):
   
    nS = len(P)
    nA = len(P[0])
    V = np.zeros(nS)
    bound = theta * (1.0 - gamma) / gamma
    iter_count = 0

    while True:
        iter_count += 1
        V_new = np.zeros(nS)
        for s in range(nS):
            q_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_s, reward, terminated in P[s][a]:
                    # When terminated, V[next_s] doesn't contribute (absorbing)
                    q_values[a] += prob * (reward + gamma * V[next_s] * (not terminated))
            V_new[s] = q_values.max()

        if np.max(np.abs(V_new - V)) < bound:
            V = V_new
            break
        V = V_new

    # Extract greedy policy
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_s, reward, terminated in P[s][a]:
                q_values[a] += prob * (reward + gamma * V[next_s] * (not terminated))
        pi[s] = int(np.argmax(q_values))

    return V, pi, iter_count


def render_policy(pi, shape=(4, 4)):
    """Render policy as arrows. 0=Left, 1=Down, 2=Right, 3=Up."""
    arrows = {0: "<", 1: "v", 2: ">", 3: "^"}
    grid = pi.reshape(shape)
    return "\n".join("  ".join(arrows[a] for a in grid[r]) for r in range(shape[0]))


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P

    gamma = 0.99
    theta = 1e-4
    V_star, pi_star, iters = value_iteration(P, gamma, theta)

    print(f"Value Iteration on FrozenLake-v1 (gamma={gamma}, theta={theta})")
    print(f"Iterations to converge: {iters}")
    print(f"\nV* table (4x4 grid):")
    print(V_star.reshape(4, 4).round(4))
    print(f"\npi* as arrows (Left=<, Down=v, Right=>, Up=^):")
    print(render_policy(pi_star))
    print(f"\npi* as integer array: {pi_star}")