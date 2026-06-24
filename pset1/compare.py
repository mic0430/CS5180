import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from vi import value_iteration
from pi import policy_iteration


def vi_backups(nS, nA, iters):
    """VI: |S| sweeps per iteration, each backup is |S|*|A| operations."""
    return iters * nS * nA * nS  # |S|^2 * |A| per iter


def pi_backups(nS, nA, iters):
    """PI per iteration: O(|S|^3) linear solve + |S|*|A| improvement."""
    return iters * (nS ** 3 + nS * nA * nS)


def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    P = env.unwrapped.P
    nS, nA = len(P), len(P[0])
    theta = 1e-4

    gammas = [0.5, 0.9, 0.99, 0.999]
    vi_iters_all, pi_iters_all = [], []

    print(f"{'gamma':>8}  {'VI iters':>10}  {'VI time':>10}  {'PI iters':>10}  {'PI time':>10}")
    print("-" * 65)

    for gamma in gammas:
        t0 = time.perf_counter()
        _, pi_vi, iters_vi = value_iteration(P, gamma, theta)
        t_vi = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, pi_pi, iters_pi = policy_iteration(P, gamma)
        t_pi = time.perf_counter() - t0

        vi_iters_all.append(iters_vi)
        pi_iters_all.append(iters_pi)

        backups_vi = vi_backups(nS, nA, iters_vi)
        backups_pi = pi_backups(nS, nA, iters_pi)

        print(f"  {gamma:6.3f}  {iters_vi:>10d}  {t_vi*1000:>8.2f}ms  "
              f"{iters_pi:>10d}  {t_pi*1000:>8.2f}ms")
        print(f"          VI backups ~{backups_vi:>12,}   PI backups ~{backups_pi:>12,}   "
              f"policies match: {np.array_equal(pi_vi, pi_pi)}")

    # Plot iteration count vs gamma
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gammas, vi_iters_all, "o-", color="#378ADD", linewidth=2,
            label="Value Iteration", markersize=8)
    ax.plot(gammas, pi_iters_all, "s-", color="#1D9E75", linewidth=2,
            label="Policy Iteration", markersize=8)
    ax.set_xlabel("Discount factor γ")
    ax.set_ylabel("Iterations to converge")
    ax.set_title("VI vs PI iteration count on FrozenLake-v1 (slippery)")
    ax.set_yscale("log")
    ax.set_xticks(gammas)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("vi_vs_pi.png", dpi=120, bbox_inches="tight")
    print(f"\nSaved figure: vi_vs_pi.png")


if __name__ == "__main__":
    main()