"""
Live policy viewer - runs a trained checkpoint in a real-time interactive
MuJoCo window. All versions (v1/v2/v3a/v3b/v4). LOCAL machine only (needs a
display).

Usage:
    python3 live_view.py <model.zip> <vecnorm.pkl> --version v4
    python3 live_view.py quadragon_v4_colab_final.zip vecnormalize_v4_colab.pkl --version v4 --speed 0.5

Controls (native MuJoCo viewer):
    - Rotate: left-drag        Pan: right-drag        Zoom: scroll
    - **Manual shove**: double-click the robot's body to select it, then
      Ctrl + right-drag to apply a force yourself, live. This is the killer
      feature for v4 testing - YOU become the perturbation and watch the
      recovery in real time.
    - Space pauses the *rendering* (physics keeps stepping in this script);
      close the window to end.

Flags:
    --speed 1.0    real-time; 0.5 = slow motion (great for reading the gait),
                   2.0 = double speed
    --stochastic   sample actions instead of deterministic (shows the policy's
                   exploration noise)

Tip for v4 policies: a v4 checkpoint has the SAME 50-dim obs as v2, so you can
run it with `--version v2` to get a world with NO automated pushes - then use
Ctrl+drag to deliver only your own, hand-timed shoves. Cleanest possible way
to judge recovery behavior.

Episodes auto-reset on fall/timeout; per-episode stats print to console.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def get_env_class(version: str):
    if version == "v1":
        from quadragon_env import QuadragonEnv
        return QuadragonEnv
    elif version == "v2":
        from quadragon_env_v2 import QuadragonEnvV2
        return QuadragonEnvV2
    elif version == "v3a":
        from quadragon_env_v3a import QuadragonEnvV3a
        return QuadragonEnvV3a
    elif version == "v3b":
        from quadragon_env_v3b import QuadragonEnvV3b
        return QuadragonEnvV3b
    elif version == "v4":
        from quadragon_env_v4 import QuadragonEnvV4
        return QuadragonEnvV4
    else:
        from quadragon_env_v5 import QuadragonEnvV5
        return QuadragonEnvV5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("vecnorm_path")
    p.add_argument("--version", choices=["v1", "v2", "v3a", "v3b", "v4", "v5", "v6"], default="v4")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed multiplier (0.5 = slow-mo)")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of deterministic")
    p.add_argument("--forward", choices=["+x", "-x", "+y", "-y"], default=None,
                   help="override the env's forward axis (v6 sign confirmation)")
    args = p.parse_args()

    EnvClass = get_env_class(args.version)
    env_kwargs = {}
    if args.forward is not None:
        if args.version == "v6":
            env_kwargs["forward_axis"] = args.forward
        else:
            print(f"note: --forward given for {args.version}; overriding display axis only")
    venv = DummyVecEnv([lambda: EnvClass(**env_kwargs)])
    venv = VecNormalize.load(args.vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model_path)
    raw_env = venv.venv.envs[0]

    # Forward vector for the overhead arrow (env attr, with display-only override)
    _AX = {"+x": np.array([1.0, 0.0]), "-x": np.array([-1.0, 0.0]),
           "+y": np.array([0.0, 1.0]), "-y": np.array([0.0, -1.0])}
    fwd2 = _AX[args.forward] if args.forward else getattr(raw_env, "_forward_vec", _AX["+x"])
    fwd_label = args.forward or getattr(raw_env, "forward_axis", "+x")

    def draw_forward_arrow(viewer):
        """Red arrow floating above the base, pointing along the forward axis."""
        base = raw_env.data.xpos[raw_env._base_id]
        start = base + np.array([0.0, 0.0, 0.12])
        end = start + np.array([fwd2[0], fwd2[1], 0.0]) * 0.18
        scn = viewer.user_scn
        scn.ngeom = 0
        g = scn.geoms[0]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
        scn.ngeom = 1

    obs = venv.reset()
    ep_num, ep_steps, ep_reward, ep_dist = 1, 0, 0.0, 0.0
    step_dt = raw_env.dt / max(args.speed, 1e-6)

    print(f"Running {args.model_path} in {args.version} env "
          f"({'stochastic' if args.stochastic else 'deterministic'}, {args.speed}x speed)")
    print(f"Forward axis: {fwd_label} (red arrow above the robot - it should point out of the HEAD)")
    print("Double-click the robot, then Ctrl + right-drag to shove it yourself.\n")

    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as viewer:
        # Frame the robot nicely on open
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20

        last_push_count = 0
        try:
            while viewer.is_running():
                t0 = time.perf_counter()

                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, r, done, info = venv.step(action)

                ep_steps += 1
                ep_reward += r[0]
                ep_dist += info[0].get("vx", 0.0) * raw_env.dt

                # Announce automated pushes (v4) as they fire
                pc = info[0].get("push_count", 0)
                if pc > last_push_count:
                    print(f"  [push #{pc} fired]")
                    last_push_count = pc

                if done[0]:
                    cause = "fell" if ep_steps < raw_env.max_episode_steps else "timeout"
                    print(f"Episode {ep_num}: {ep_steps} steps ({cause}), "
                          f"reward {ep_reward:.1f}, distance {ep_dist:.2f} m")
                    ep_num += 1
                    ep_steps, ep_reward, ep_dist = 0, 0.0, 0.0
                    last_push_count = 0
                    # DummyVecEnv auto-reset already gave us the fresh obs

                draw_forward_arrow(viewer)
                viewer.sync()

                # Real-time pacing (scaled by --speed)
                elapsed = time.perf_counter() - t0
                if elapsed < step_dt:
                    time.sleep(step_dt - elapsed)
        except KeyboardInterrupt:
            pass

    print("Viewer closed.")
    venv.close()


if __name__ == "__main__":
    main()
