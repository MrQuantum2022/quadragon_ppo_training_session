"""
Smoke test for QuadragonEnv (v1) / QuadragonEnvV2 - run before any training.

Usage:
    python3 smoke_test.py --version v1
    python3 smoke_test.py --version v2

Checks: API compliance, obs shape/sanity, zero-action stability,
random-action rollout, termination triggers, seeded determinism,
prev_action stored in raw [-1,1] space. v2 additionally checks the
foot-state observation dims and that r_clearance appears in info.
"""

import argparse
import numpy as np


def get_env(version: str):
    if version == "v1":
        from quadragon_env import QuadragonEnv
        return QuadragonEnv(), 42
    else:
        from quadragon_env_v2 import QuadragonEnvV2
        return QuadragonEnvV2(), 50


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2"], default="v1")
    args = p.parse_args()

    env, obs_dim = get_env(args.version)
    print(f"Testing {args.version} (expected obs dim {obs_dim})\n")

    # --- 1. Reset & obs sanity ---
    obs, info = env.reset(seed=42)
    assert obs.shape == (obs_dim,), f"obs shape {obs.shape}, expected ({obs_dim},)"
    assert not np.any(np.isnan(obs)), "NaN in initial obs"
    print(f"[1] reset OK - obs shape {obs.shape}, "
          f"range [{obs.min():.3f}, {obs.max():.3f}]")

    # --- 2. Zero action: should just stand there, full episode ---
    obs, _ = env.reset(seed=0)
    steps, total_r = 0, 0.0
    for _ in range(env.max_episode_steps):
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        total_r += r
        steps += 1
        if term:
            break
    status = "survived (truncated)" if not term else f"FELL at step {steps}"
    print(f"[2] zero-action: {status}, {steps} steps, "
          f"final height {info['height']:.3f}, total reward {total_r:.2f}")
    assert not term, "Robot fell over doing nothing - check reset settle or thresholds"

    # --- 3. Random actions: expect survival OR legitimate termination, no NaN ---
    obs, _ = env.reset(seed=1)
    steps = 0
    for _ in range(env.max_episode_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert not np.any(np.isnan(obs)), f"NaN in obs at step {steps}"
        steps += 1
        if term or trunc:
            break
    print(f"[3] random actions: ran {steps} steps, "
          f"terminated={term}, truncated={trunc}, height {info['height']:.3f}")

    # --- 4. Seeded determinism ---
    def rollout(seed, n=50):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        traj = [obs.copy()]
        for _ in range(n):
            a = rng.uniform(-1, 1, 12).astype(np.float32)
            obs, *_ = env.step(a)
            traj.append(obs.copy())
        return np.array(traj)

    t1, t2 = rollout(7), rollout(7)
    max_diff = np.abs(t1 - t2).max()
    print(f"[4] determinism: max obs diff across identical seeded rollouts = {max_diff:.2e}")
    assert max_diff < 1e-10, "Non-deterministic under identical seed"

    # --- 5. prev_action stored in raw [-1,1] space (same index in v1 and v2 -
    #     v2 appends foot state AFTER these 42, so this slice doesn't shift) ---
    obs, _ = env.reset(seed=3)
    test_a = np.full(12, 0.5, dtype=np.float32)
    obs, *_ = env.step(test_a)
    stored = obs[30:42]
    assert np.allclose(stored, 0.5), f"prev_action in obs is {stored[0]}, expected 0.5 (raw [-1,1] space)"
    print(f"[5] prev_action stored in raw [-1,1] space: confirmed ({stored[0]:.2f})")

    # --- 6. v2-only: foot state present in obs, r_clearance present in info ---
    if args.version == "v2":
        obs, _ = env.reset(seed=5)
        contact = obs[42:46]
        heights = obs[46:50]
        assert np.all((contact == 0.0) | (contact == 1.0)), f"contact flags not binary: {contact}"
        print(f"[6] foot contact flags at reset: {contact}  (heights scaled: {heights})")

        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        assert "r_clearance" in info, "r_clearance missing from info - v2 reward not wired up"
        print(f"[6b] r_clearance present: {info['r_clearance']:.4f}, n_swing={info['n_swing']}")

    env.close()
    print(f"\nAll smoke tests passed for {args.version}.")


if __name__ == "__main__":
    main()
