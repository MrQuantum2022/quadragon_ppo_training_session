"""
Eval several checkpoints and report reward + final velocity for each,
so you can pick the genuine best (not the final) and see where the
collapse starts. Run in Colab where the checkpoints/ folder lives.
"""
import glob, re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from quadragon_env import QuadragonEnv as EnvClass

RUN = "v1_fixed_colab"

ckpts = glob.glob(f"checkpoints/v1_fixed/quadragon_{RUN}_*_steps.zip")
ckpts = [c for c in ckpts if "vecnormalize" not in c]
ckpts.sort(key=lambda p: int(re.search(r"_(\d+)_steps", p).group(1)))

print(f"{'step':>10} | {'mean_reward':>11} | {'final_vx':>8} | {'survived':>8}")
print("-" * 48)
for c in ckpts:
    step = int(re.search(r"_(\d+)_steps", c).group(1))
    vecnorm = f"checkpoints/v1_fixed/quadragon_{RUN}_vecnormalize_{step}_steps.pkl"
    venv = DummyVecEnv([lambda: EnvClass()])
    venv = VecNormalize.load(vecnorm, venv)
    venv.training = False
    venv.norm_reward = False
    model = PPO.load(c)

    # 3 episodes, average
    rewards, vxs, survived = [], [], []
    for ep in range(3):
        obs = venv.reset()
        total, last_vx = 0.0, 0.0
        for step_i in range(500):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = venv.step(a)
            total += r[0]
            last_vx = info[0]["vx"]
            if done[0]:
                break
        rewards.append(total)
        vxs.append(last_vx)
        survived.append(step_i + 1)
    print(f"{step:>10} | {np.mean(rewards):>11.1f} | {np.mean(vxs):>8.3f} | {np.mean(survived):>8.0f}")