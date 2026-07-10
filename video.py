import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
VERSION = 'v2'   # 'v1' or 'v2'
video_version= 'v2_clearance_fixed'  # 'v1_fixed' or 'v2'
if VERSION == 'v2':
    from quadragon_env_v2 import QuadragonEnvV2 as EnvClass
else:
    from quadragon_env import QuadragonEnv as EnvClass

MODEL_PATH = 'checkpoints/v2_clearance_fixed/quadragon_v2_clearance_fixed_colab_2900000_steps.zip'      # or a checkpoints/ file
VECNORM_PATH = 'checkpoints/v2_clearance_fixed/quadragon_v2_clearance_fixed_colab_vecnormalize_2900000_steps.pkl'

venv = DummyVecEnv([lambda: EnvClass(render_mode='rgb_array')])
venv = VecNormalize.load(VECNORM_PATH, venv)
venv.training = False        # freeze stats - deployment-identical inference
venv.norm_reward = False

model = PPO.load(MODEL_PATH)
raw_env = venv.venv.envs[0]

obs = venv.reset()
frames, total_r = [], 0.0
for step in range(500):   # 10s at 50Hz
    action, _ = model.predict(obs, deterministic=True)
    obs, r, done, info = venv.step(action)
    total_r += r[0]
    if step % 2 == 0:      # 25fps video
        frames.append(raw_env.render())
    if done[0]:
        print(f'Episode ended at step {step}')
        break

print(f'Total reward: {total_r:.1f}, final vx: {info[0]["vx"]:.3f} m/s')
if VERSION == 'v2':
    print(f'(v2 also tracked: r_clearance last step = {info[0].get("r_clearance", "n/a")})')
imageio.mimsave(f'rollout{video_version}.mp4', frames, fps=25)

from IPython.display import Video
Video(f'rollout{video_version}.mp4', embed=True, width=640)