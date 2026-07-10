"""
Quadragon v2 - foot clearance (first incremental step past v1).

Subclasses QuadragonEnv so v1 stays intact and runnable as the baseline.
The ONLY additions here:
  1. Foot-clearance reward: during swing (foot not in floor contact), reward
     lifting the foot toward a target height; this directly targets the
     "dragging / newborn flap" gait v1 produced.
  2. Foot state appended to observation (4 contact flags + 4 heights = 8 dims),
     so the policy can actually perceive what the reward is now scoring.

Everything else - action space, physics, termination, all v1 reward terms -
is inherited unchanged. Per SPEC.md v2 discipline: ONE new term, then retrain
and re-check before adding phase timing.

Obs: 42 (v1) + 8 (foot state) = 50 dims.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
import mujoco

from quadragon_env import QuadragonEnv


class QuadragonEnvV2(QuadragonEnv):
    def __init__(self, foot_clearance_target: float = 0.03, w_clearance: float = 0.4,
                 max_swing_duration_s: float = 0.4, **kwargs):
        super().__init__(**kwargs)

        # Foot geoms defined in the model (calf-tip spheres)
        foot_names = ["FR_foot", "BR_foot", "BL_foot", "FL_foot"]
        self._foot_gids = np.array([self.model.geom(n).id for n in foot_names])
        self._floor_gid = self.model.geom("floor").id
        self._foot_radius = float(self.model.geom(self._foot_gids[0]).size[0])

        # Exploit fix: a foot held motionless near the target height earns max
        # clearance reward forever, with no requirement it ever actually plants.
        # Cap continuous airborne time - beyond this, clearance reward for that
        # foot is cut to zero until it touches down again. 0.4s is a generous
        # upper bound on a real single-leg swing phase for this platform.
        self.max_swing_duration_s = max_swing_duration_s
        self._swing_duration = np.zeros(4)

        # Target height for the foot SPHERE CENTER during swing. At rest the
        # center sits at ~= radius (sphere resting on floor). Target is radius +
        # desired clearance, so we're asking the foot bottom to clear the ground
        # by roughly `foot_clearance_target` meters.
        self.foot_clearance_target = self._foot_radius + foot_clearance_target
        self.w_clearance = w_clearance

        # Extend observation space by 8 (4 contact flags + 4 heights)
        obs_dim = self.observation_space.shape[0] + 8
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ---- Foot sensing ----

    def _foot_heights(self) -> np.ndarray:
        return np.array([self.data.geom_xpos[g][2] for g in self._foot_gids])

    def _feet_contact(self) -> np.ndarray:
        flags = {int(g): False for g in self._foot_gids}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if self._floor_gid in pair:
                for g in self._foot_gids:
                    if int(g) in pair:
                        flags[int(g)] = True
        return np.array([flags[int(g)] for g in self._foot_gids], dtype=np.float32)

    # ---- Obs: v1 obs + foot state ----

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        contact = self._feet_contact()
        heights = self._foot_heights() * 10.0   # scale ~O(0.02-0.05) into a friendlier range
        return np.concatenate([base, contact, heights]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._swing_duration = np.zeros(4)   # must reset per-episode, not carry over
        return obs, info

    # ---- Step: inherit everything, add clearance reward ----

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        contact = self._feet_contact()
        heights = self._foot_heights()

        # Track continuous airborne time per foot - resets to 0 the instant a
        # foot touches down, increments by dt every step it stays up.
        swing = (contact < 0.5)
        self._swing_duration = np.where(swing, self._swing_duration + self.dt, 0.0)

        # Reward clearance only for feet that are (a) currently swinging AND
        # (b) haven't overstayed a plausible swing duration. Condition (b) is
        # the exploit fix: without it, a foot held motionless near the target
        # height forever earns max reward forever, with no requirement it ever
        # actually completes a step and plants again.
        eligible = swing & (self._swing_duration <= self.max_swing_duration_s)
        if np.any(eligible):
            err = heights[eligible] - self.foot_clearance_target
            clearance_score = np.exp(-((err / 0.02) ** 2))  # peak at target, falls off
            r_clearance = self.w_clearance * float(np.mean(clearance_score))
        else:
            r_clearance = 0.0   # all feet planted, or all overstayed swing - nothing to reward

        reward += r_clearance
        info["r_clearance"] = r_clearance
        info["n_swing"] = int(np.sum(swing))
        return obs, reward, terminated, truncated, info
