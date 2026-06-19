import numpy as np
import matplotlib.pyplot as plt

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy


def run_episodes(env_factory, policy_factory, N, seed_base=0):
    returns = np.empty(N, dtype=np.float64)
    rejected_all = 0
    for i in range(N):
        env = env_factory(seed_base + i)
        policy = policy_factory(seed_base + i)
        obs, info = env.reset(seed=seed_base + i)
        done = False
        total = 0.0
        accepted = False
        while not done:
            a = policy.act(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            if a == 1:
                accepted = True
            done = term or trunc
        returns[i] = total
        if not accepted:
            rejected_all += 1
    return returns, rejected_all / N


def summarize(name, returns, frac_rej):
    mean = returns.mean()
    se = returns.std(ddof=1) / np.sqrt(len(returns))
    print(f"  {name:<22} mean={mean:.4f}  SE={se:.4f}  frac_reject_all={frac_rej:.4f}")
    return mean, se


def part_c(N=10_000, T=4, K=4):
    print("=" * 60)
    print(f"Part (c): N={N}, T={T}, K={K}, sigma=0")
    print("=" * 60)

    def env_factory(seed):
        return ApartmentEnv(T=T, K=K, seed=seed, noise_std=0.0)

    results = {}
    rets, fr = run_episodes(env_factory, lambda s: RandomPolicy(T=T, seed=s), N)
    summarize("RandomPolicy", rets, fr)
    results["Random"] = rets

    print("\n  ThresholdPolicy sweep:")
    best_umin, best_mean, best_rets = None, -np.inf, None
    for u_min in [1, 2, 3, 4]:
        rets, fr = run_episodes(env_factory, lambda s, u=u_min: ThresholdPolicy(u_min=u), N)
        mean, se = summarize(f"ThresholdPolicy({u_min})", rets, fr)
        if mean > best_mean:
            best_mean, best_umin, best_rets = mean, u_min, rets
    print(f"  --> Best fixed threshold: u_min = {best_umin}  (mean = {best_mean:.4f})")
    results[f"Threshold(u_min={best_umin})"] = best_rets

    print()
    rets, fr = run_episodes(env_factory, lambda s: OptimalPolicy(), N)
    summarize("OptimalPolicy", rets, fr)
    results["Optimal"] = rets

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    names = ["Random", f"Threshold(u_min={best_umin})", "Optimal"]
    colors = ["#888780", "#378ADD", "#1D9E75"]
    bins = np.arange(-0.5, 5.5, 1.0)
    for ax, name, color in zip(axes, names, colors):
        ax.hist(results[name], bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{name}\nmean = {results[name].mean():.3f}")
        ax.set_xlabel("Return (utility)")
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Episode count")
    fig.suptitle(f"Return distributions (N={N}, T={T}, K={K})", y=1.02)
    fig.tight_layout()
    fig.savefig("returns_hist.png", dpi=120, bbox_inches="tight")
    print(f"\n  Saved figure: returns_hist.png")

    return best_umin


def part_d(N=10_000, T=4, K=4, sigmas=(0.0, 0.5, 1.0, 2.0)):
    print("\n" + "=" * 60)
    print(f"Part (d): noise robustness, sigmas = {sigmas}")
    print("=" * 60)

    means = {"Random": [], "Threshold(3)": [], "Optimal": []}
    ses = {"Random": [], "Threshold(3)": [], "Optimal": []}

    for sigma in sigmas:
        print(f"\n  sigma = {sigma}:")
        def env_factory(seed, s=sigma):
            return ApartmentEnv(T=T, K=K, seed=seed, noise_std=s)

        rets, fr = run_episodes(env_factory, lambda s: RandomPolicy(T=T, seed=s), N)
        m, e = summarize("RandomPolicy", rets, fr)
        means["Random"].append(m); ses["Random"].append(e)

        rets, fr = run_episodes(env_factory, lambda s: ThresholdPolicy(u_min=3), N)
        m, e = summarize("ThresholdPolicy(3)", rets, fr)
        means["Threshold(3)"].append(m); ses["Threshold(3)"].append(e)

        rets, fr = run_episodes(env_factory, lambda s: OptimalPolicy(), N)
        m, e = summarize("OptimalPolicy", rets, fr)
        means["Optimal"].append(m); ses["Optimal"].append(e)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(sigmas)
    colors = {"Random": "#888780", "Threshold(3)": "#378ADD", "Optimal": "#1D9E75"}
    for name, ys in means.items():
        ax.errorbar(xs, ys, yerr=ses[name], marker="o", capsize=4,
                    color=colors[name], label=name, linewidth=2)
    ax.set_xlabel("Noise std deviation sigma")
    ax.set_ylabel("Mean return (utility)")
    ax.set_title("Policy robustness to observation noise")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("noise_robustness.png", dpi=120, bbox_inches="tight")
    print(f"\n  Saved figure: noise_robustness.png")


if __name__ == "__main__":
    np.random.seed(42)
    part_c(N=10_000)
    part_d(N=10_000)