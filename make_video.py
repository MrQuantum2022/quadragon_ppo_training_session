"""
Local rollout video generator - handles v1 / v2 / v3a / v3b.

Runs on your LOCAL machine (Colab's headless renderer errors on video export).
Loads a trained checkpoint + its VecNormalize stats, runs a deterministic
rollout, and writes an mp4.

Usage:
    python3 make_video.py <model.zip> <vecnorm.pkl> --version v3b
    python3 make_video.py quadragon_v2_baseline.zip vecnormalize_v2_baseline.pkl --version v2 --out v2.mp4

Deps (local):
    pip install imageio imageio-ffmpeg

Notes:
    - --version MUST match how the checkpoint was trained (v3b's obs is 52-dim;
      loading it as v2 would shape-mismatch immediately).
    - Uses the exact eval-time VecNormalize handling (training=False,
      norm_reward=False) so what you watch matches deployment behavior.
    - Prints the same headline stats (reward, velocity, distance) so the video
      filename and the numbers stay linked in your notes.
"""

import argparse
import os

import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Headless-friendly GL backend. If you have a display and want a live window
# instead, this still works for offscreen render. egl/osmesa both fine.
os.environ.setdefault("MUJOCO_GL", "egl")


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
    elif version == "v5":
        from quadragon_env_v5 import QuadragonEnvV5
        return QuadragonEnvV5
    else:
        from quadragon_env_v6 import QuadragonEnvV6
        return QuadragonEnvV6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("vecnorm_path")
    p.add_argument("--version", choices=["v1", "v2", "v3a", "v3b", "v4", "v5", "v6"], default="v2")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = args.out or f"rollout_{args.version}.mp4"

    EnvClass = get_env_class(args.version)
    venv = DummyVecEnv([lambda: EnvClass(render_mode="rgb_array")])
    venv = VecNormalize.load(args.vecnorm_path, venv)
    venv.training = False       # freeze normalization stats
    venv.norm_reward = False    # report raw reward

    model = PPO.load(args.model_path)
    raw_env = venv.venv.envs[0]

    obs = venv.reset()
    frames, total_r = [], 0.0
    last_info = None
    render_every = max(1, round((1.0 / raw_env.dt) / args.fps))  # subsample to target fps

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(action)
        total_r += r[0]
        last_info = info[0]
        if step % render_every == 0:
            # Best-effort forward arrow in offscreen frames (renderer scene
            # access varies by mujoco version; skip silently if unavailable)
            try:
                import mujoco as _mj
                fwd2 = getattr(raw_env, "_forward_vec", np.array([1.0, 0.0]))
                scn = raw_env._renderer.scene if raw_env._renderer else None
                if scn is not None and scn.ngeom < scn.maxgeom:
                    base = raw_env.data.xpos[raw_env._base_id]
                    start = base + np.array([0.0, 0.0, 0.12])
                    end = start + np.array([fwd2[0], fwd2[1], 0.0]) * 0.18
                    g = scn.geoms[scn.ngeom]
                    _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                     np.zeros(9), np.array([1.0, 0.15, 0.15, 1.0], dtype=np.float32))
                    _mj.mjv_connector(g, _mj.mjtGeom.mjGEOM_ARROW, 0.012, start, end)
                    scn.ngeom += 1
            except Exception:
                pass
            frames.append(raw_env.render())
        if done[0]:
            print(f"Episode ended at step {step}")
            break

    vx = last_info.get("vx", float("nan")) if last_info else float("nan")
    print(f"Total reward: {total_r:.1f}, final vx: {vx:.3f} m/s, frames: {len(frames)}")

    imageio.mimsave(out_path, frames, fps=args.fps)
    print(f"Saved {out_path}")
    venv.close()


if __name__ == "__main__":
    main()
