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
    elif version == "v2":
        from quadragon_env_v2 import QuadragonEnvV2
        return QuadragonEnvV2(), 50
    elif version == "v3a":
        from quadragon_env_v3a import QuadragonEnvV3a
        return QuadragonEnvV3a(), 50   # data-driven arm adds no obs dims
    elif version == "v3b":
        from quadragon_env_v3b import QuadragonEnvV3b
        return QuadragonEnvV3b(), 52   # phase-clock arm adds sin/cos of phase
    elif version == "v4":
        from quadragon_env_v4 import QuadragonEnvV4
        return QuadragonEnvV4(), 50    # pushes add no obs dims
    elif version == "v5":
        from quadragon_env_v5 import QuadragonEnvV5
        return QuadragonEnvV5(), 50    # servo model adds no obs dims
    else:
        from quadragon_env_v6 import QuadragonEnvV6
        return QuadragonEnvV6(), 50    # direction change adds no obs dims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2", "v3a", "v3b", "v4", "v5", "v6"], default="v1")
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
    if args.version in ("v4", "v5", "v6"):
        # v4/v5/v6's whole design is that passivity is no longer viable: random pushes
        # should threaten a do-nothing robot. Falling here is EXPECTED (proves
        # the pushes are real); surviving just means this seed drew mild pushes.
        # Either outcome passes - we only report which happened.
        print("    (v4: falling under pushes while passive is expected, not a failure)")
    else:
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

    # --- 6. Foot state + clearance: present in v2 and everything built on it ---
    if args.version in ("v2", "v3a", "v3b", "v4", "v5", "v6"):
        obs, _ = env.reset(seed=5)
        contact = obs[42:46]
        heights = obs[46:50]
        assert np.all((contact == 0.0) | (contact == 1.0)), f"contact flags not binary: {contact}"
        print(f"[6] foot contact flags at reset: {contact}  (heights scaled: {heights})")

        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        assert "r_clearance" in info, "r_clearance missing from info - clearance reward not wired up"
        print(f"[6b] r_clearance present: {info['r_clearance']:.4f}, n_swing={info['n_swing']}")

    # --- 7. v3-specific reward terms + (v3b) appended phase dims ---
    if args.version == "v3a":
        obs, _ = env.reset(seed=6)
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        assert "r_balance" in info and "r_freq" in info, "v3a coordination terms missing from info"
        print(f"[7] v3a terms present: r_balance={info['r_balance']:.4f}, "
              f"r_freq={info['r_freq']:.4f}, duty_std={info['duty_std']:.3f}")

    if args.version == "v3b":
        obs, _ = env.reset(seed=6)
        phase0 = obs[50:52].copy()   # sin/cos appended after the 50 v2 dims
        obs, r, term, trunc, info = env.step(np.zeros(12, dtype=np.float32))
        phase1 = obs[50:52]
        assert "r_phase" in info, "v3b phase reward missing from info"
        assert not np.allclose(phase0, phase1), "phase clock not advancing"
        print(f"[7] v3b phase present: r_phase={info['r_phase']:.4f}, "
              f"phase sin/cos advanced {phase0} -> {phase1}")

    if args.version == "v5":
        obs, _ = env.reset(seed=6)
        obs, r, term, trunc, info = env.step(np.full(12, 0.9, dtype=np.float32))
        assert "servo_lag" in info, "v5 servo model not wired up"
        assert info["servo_lag"] > 0.01, "servo_lag ~0 on a large command - lag/slew not applying"
        print(f"[7] v5 servo model active: servo_lag={info['servo_lag']:.3f} rad on a large step command")

    if args.version == "v6":
        from quadragon_env_v6 import QuadragonEnvV6
        # Axis wiring check: identical seeded rollouts under '+x' vs '+y' must
        # report DIFFERENT forward velocities (same physics, different projection)
        vs = {}
        for ax in ("+x", "+y"):
            e = QuadragonEnvV6(forward_axis=ax)
            e.reset(seed=11)
            rng = np.random.default_rng(11)
            tot = 0.0
            for _ in range(60):
                _, _, _, _, inf = e.step(rng.uniform(-1, 1, 12).astype(np.float32))
                tot += inf["vx"]
            vs[ax] = tot
            e.close()
        assert abs(vs["+x"] - vs["+y"]) > 1e-6, "forward axis not affecting reported vx"
        print(f"[7] v6 axis wiring: cumulative vx  +x={vs['+x']:+.3f}  +y={vs['+y']:+.3f}  (differ: OK)")

    env.close()
    print(f"\nAll smoke tests passed for {args.version}.")


if __name__ == "__main__":
    main()
