from env import ApartmentEnv
from policies import RandomPolicy


def main():
    T, K = 4, 4
    env = ApartmentEnv(T=T, K=K, seed=0)
    policy = RandomPolicy(T=T, seed=0)

    obs, info = env.reset(seed=0)
    print(f"{'t':>3}  {'U_t':>4}  {'action':>7}  {'reward':>7}  {'done':>5}")
    print("-" * 36)
    done = False
    while not done:
        t = int(obs[0])
        U_t = info["true_U"]
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        a_name = "accept" if action == 1 else "reject"
        print(f"{t:>3}  {U_t:>4}  {a_name:>7}  {reward:>7.2f}  {str(done):>5}")


if __name__ == "__main__":
    main()