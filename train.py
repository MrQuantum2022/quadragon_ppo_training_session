"""
PPO training for Quadragon (v1 or v2).

Usage:
    python3 train.py --version v1 --n-envs 8 --timesteps 3000000
    python3 train.py --version v2 --n-envs 8 --timesteps 3000000

Monitor:
    tensorboard --logdir runs/

--version selects the env:
    v1 -> QuadragonEnv       (42-dim obs, minimal reward)  [baseline, DONE]
    v2 -> QuadragonEnvV2     (50-dim obs, + foot clearance) [current step]

target_kl=0.02 is baked in from the start here - it was the fix that stopped
v1's clip_fraction/approx_kl runaway, so v2 gets it from step 0 rather than
needing a mid-run patch. SB3 infers obs/action dims from the env, so the same
script handles both versions with no shape edits.

Checkpoints -> checkpoints/, final model -> quadragon_<run-name>_final.zip
(+ its matching vecnormalize_<run-name>.pkl - BOTH needed for deployment/eval).
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from quadragon_env import QuadragonEnv
from quadragon_env_v2 import QuadragonEnvV2
from quadragon_env_v3a import QuadragonEnvV3a
from quadragon_env_v3b import QuadragonEnvV3b

ENVS = {"v1": QuadragonEnv, "v2": QuadragonEnvV2,
        "v3a": QuadragonEnvV3a, "v3b": QuadragonEnvV3b}


def make_env(env_cls, seed: int):
    def _init():
        env = Monitor(env_cls())
        env.reset(seed=seed)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v1", "v2", "v3a", "v3b"], default="v2")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    args = p.parse_args()

    run_name = args.run_name or f"{args.version}_colab"
    env_cls = ENVS[args.version]

    if args.n_envs == 1:
        venv = DummyVecEnv([make_env(env_cls, args.seed)])
    else:
        venv = SubprocVecEnv([make_env(env_cls, args.seed + i) for i in range(args.n_envs)])

    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.02,   # the v1 fix, applied from the start for v2
        seed=args.seed,
        verbose=1,
        tensorboard_log="runs",
    )

    ckpt = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path="checkpoints",
        name_prefix=f"quadragon_{run_name}",
        save_vecnormalize=True,
    )

    print(f"Training {args.version} (obs dim {venv.observation_space.shape[0]}) "
          f"as run '{run_name}', {args.n_envs} envs, {args.timesteps} steps")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, tb_log_name=run_name)

    model.save(f"quadragon_{run_name}_final")
    venv.save(f"vecnormalize_{run_name}.pkl")
    print(f"Saved quadragon_{run_name}_final.zip and vecnormalize_{run_name}.pkl")
    print("Deployment/eval note: BOTH files required - policy is meaningless "
          "without its matching normalization statistics.")


if __name__ == "__main__":
    main()
