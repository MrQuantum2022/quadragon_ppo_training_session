"""
Resume training from the latest checkpoint, with target_kl added as a
safety valve against the runaway clip_fraction/approx_kl seen at 491k-852k
steps (0.385->0.448 and 0.041->0.055 - trending the wrong way, not settling).

target_kl makes PPO stop taking further epoch updates on a rollout batch
once the KL divergence exceeds the threshold, instead of blindly running
all n_epochs regardless of how far the policy has already drifted. This
is the standard fix for exactly this symptom.

Run this from Colab in place of a fresh train.py call.
"""

import glob
import re

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from quadragon_env import QuadragonEnv

RUN_NAME = "v1_colab"
N_ENVS = 8
TOTAL_TIMESTEPS = 3_000_000
TARGET_KL = 0.02   # standard default; PPO stops epoch updates early past this


def make_env(seed: int):
    def _init():
        env = Monitor(QuadragonEnv())
        env.reset(seed=seed)
        return env
    return _init


def find_latest_checkpoint(run_name: str):
    pattern = f"checkpoints/quadragon_{run_name}_*_steps.zip"
    candidates = glob.glob(pattern)
    # Exclude the vecnormalize_*.pkl files' zip cousins accidentally matching
    candidates = [c for c in candidates if "vecnormalize" not in c]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    candidates.sort(key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))
    latest_model = candidates[-1]
    step = int(re.search(r"_(\d+)_steps", latest_model).group(1))
    latest_vecnorm = f"checkpoints/quadragon_{run_name}_vecnormalize_{step}_steps.pkl"
    return latest_model, latest_vecnorm, step


if __name__ == "__main__":
    latest_model, latest_vecnorm, step = find_latest_checkpoint(RUN_NAME)
    print(f"Resuming from step {step}")
    print(f"  model:     {latest_model}")
    print(f"  vecnorm:   {latest_vecnorm}")

    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecNormalize.load(latest_vecnorm, venv)
    venv.training = True

    model = PPO.load(latest_model, env=venv)
    model.target_kl = TARGET_KL
    print(f"target_kl set to {TARGET_KL} - this is the fix")

    remaining = TOTAL_TIMESTEPS - step
    model.learn(total_timesteps=remaining, reset_num_timesteps=False, tb_log_name=RUN_NAME)

    model.save(f"quadragon_{RUN_NAME}_final")
    venv.save(f"vecnormalize_{RUN_NAME}.pkl")
    print(f"Saved quadragon_{RUN_NAME}_final.zip and vecnormalize_{RUN_NAME}.pkl")
