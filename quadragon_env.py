"""
Quadragon v1 Gymnasium environment - minimal viable walking.

Wraps quadragon_mg995.xml (MuJoCo, MG995-corrected torque limits).
Design follows SPEC.md: deliberately simple - no phase variable, no contact
terms, no schedulers. The bar for v1 is "walks forward without falling",
not biological realism. Sophistication returns in v2 once this converges.

Observation (42 dims), all hardware-obtainable for sim-to-real later:
    [ 0:12]  joint positions, normalized to [-1, 1] via each joint's ctrlrange
    [12:24]  joint velocities, scaled (rad/s * 0.1)
    [24:27]  projected gravity in base frame (what an IMU accelerometer gives
             you at rest; replaces raw orientation quat - better sim-to-real)
    [27:30]  base angular velocity in base frame, scaled (gyro-equivalent)
    [30:42]  previous action, in RAW [-1, 1] policy-output space
             (NOT radians - this is the Phase 1 bug fixed by construction)

Action (12 dims): [-1, 1] per actuator, decoded linearly into that
actuator's ctrlrange. Policy never touches radians directly.

Forward direction: +X in world frame (verified: front hips at x=+0.04,
back hips at x=-0.06).

Control rate: physics at 500 Hz (dt=0.002), decimation 10 -> policy acts
at 50 Hz, matching the PCA9685's 50 Hz servo PWM on the real robot.
"""

from __future__ import annotations

import os

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class QuadragonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | None = None,
        target_velocity: float = 0.15,   # m/s - modest goal for a small sprawler
        episode_duration_s: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__()

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "quadragon_mg995.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self._renderer = None

        # --- Control timing ---
        self.decimation = 10                        # physics steps per env step
        self.dt = self.model.opt.timestep * self.decimation   # 0.02s -> 50 Hz
        self.max_episode_steps = int(episode_duration_s / self.dt)

        # --- Task ---
        self.target_velocity = target_velocity

        # --- Actuator control ranges (per-joint decode maps) ---
        self.n_act = self.model.nu                  # 12
        ctrlrange = self.model.actuator_ctrlrange.copy()   # (12, 2)

        # action=0 must map to each joint's TRUE mechanical rest angle (CAD-neutral,
        # qpos=0), not the arithmetic midpoint of ctrlrange. For symmetric ranges
        # (hip: +/-0.698) these coincide, so this is a no-op there. For the
        # deliberately asymmetric calf range (-1.571 to 0.175, a real one-directional
        # knee bend), the midpoint is -0.698 rad - a 42 degree bias off actual rest.
        # Fixed here: scale asymmetrically on either side of the true zero instead.
        self._natural_zero = np.zeros(self.n_act)
        self._ctrl_pos_scale = ctrlrange[:, 1] - self._natural_zero   # action=+1 -> upper bound
        self._ctrl_neg_scale = self._natural_zero - ctrlrange[:, 0]   # action=-1 -> lower bound

        # Map actuator i -> qpos/qvel address of its joint (skip freejoint's 7/6)
        self._act_qpos_adr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.n_act)
        ])
        self._act_qvel_adr = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.n_act)
        ])

        self._base_id = self.model.body("base_link").id

        # --- Termination thresholds ---
        # Verified settled standing height is 0.132 m; terminate well below it.
        self.min_height = 0.06
        # Projected gravity z is -1.0 when perfectly upright, 0 at 90 deg tilt.
        # -0.5 corresponds to ~60 deg of tilt - unrecoverable for this platform.
        self.max_tilt_gz = -0.5

        # --- Reward weights (v1 minimal set) ---
        self.w_forward = 1.0
        self.w_upright = 0.5
        self.w_action_rate = 0.25
        self.w_torque = 0.001
        self.alive_bonus = 0.05

        # --- Spaces ---
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_act,), dtype=np.float32)
        obs_dim = 12 + 12 + 3 + 3 + 12
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_action = np.zeros(self.n_act, dtype=np.float32)
        self._step_count = 0

        # Forward direction (world frame). Legacy default '+x' - v6 discovered
        # the body's true long axis is Y and overrides this. One source of
        # truth: env versions and viewer tools all read forward from here.
        self.forward_axis = "+x"
        self._forward_vec = np.array([1.0, 0.0])

    def _forward_velocity(self) -> float:
        """Base velocity along the configured forward axis (world frame)."""
        return float(np.dot(self.data.qvel[:2], self._forward_vec))

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: np.ndarray) -> np.ndarray:
        """[-1,1] policy output -> per-joint target angle.
        action=0 -> joint's true rest angle (0.0). Positive/negative actions
        scale independently toward that side's ctrlrange bound, so asymmetric
        ranges (e.g. calf) are handled correctly without biasing the zero point."""
        action = np.clip(action, -1.0, 1.0)
        return self._natural_zero + np.where(
            action >= 0, action * self._ctrl_pos_scale, action * self._ctrl_neg_scale
        )

    def _base_rotmat(self) -> np.ndarray:
        """3x3 rotation matrix of base_link (world <- base)."""
        return self.data.xmat[self._base_id].reshape(3, 3)

    def _projected_gravity(self) -> np.ndarray:
        """World gravity direction expressed in the base frame.
        Upright and level -> [0, 0, -1]."""
        R = self._base_rotmat()
        return R.T @ np.array([0.0, 0.0, -1.0])

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._act_qpos_adr]
        qvel = self.data.qvel[self._act_qvel_adr]

        # Same asymmetric normalization as the action decode, for consistency -
        # a joint sitting exactly at rest reads as 0.0 in the observation too.
        qc = qpos - self._natural_zero
        joint_pos_n = np.where(qc >= 0, qc / self._ctrl_pos_scale, qc / self._ctrl_neg_scale)
        joint_vel_n = qvel * 0.1

        grav = self._projected_gravity()

        # Angular velocity in base frame (freejoint qvel[3:6] is world-frame)
        R = self._base_rotmat()
        ang_vel_base = R.T @ self.data.qvel[3:6]
        ang_vel_n = ang_vel_base * 0.25

        return np.concatenate([
            joint_pos_n,
            joint_vel_n,
            grav,
            ang_vel_n,
            self._prev_action,
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Small random joint perturbation so the policy doesn't overfit to
        # one exact start state (+/- ~3 degrees).
        noise = self.np_random.uniform(-0.05, 0.05, size=self.n_act)
        self.data.qpos[self._act_qpos_adr] += noise

        mujoco.mj_forward(self.model, self.data)

        # Hold current pose, let it settle onto its feet for 0.5s so
        # episodes start from a physically consistent standing state.
        self.data.ctrl[:] = self.data.qpos[self._act_qpos_adr]
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(self.n_act, dtype=np.float32)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = self._decode_action(action)

        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # --- Reward terms ---
        # Forward velocity along +X, rewarded up to target then flat
        # (no bonus for overspeeding - keeps gait controlled).
        vx = self._forward_velocity()
        r_forward = self.w_forward * min(vx, self.target_velocity) / self.target_velocity

        grav = self._projected_gravity()
        r_upright = -self.w_upright * float(grav[0] ** 2 + grav[1] ** 2)

        da = action - self._prev_action
        r_action_rate = -self.w_action_rate * float(np.mean(da ** 2))

        r_torque = -self.w_torque * float(np.mean(self.data.actuator_force ** 2))

        reward = r_forward + r_upright + r_action_rate + r_torque + self.alive_bonus

        # Update prev_action BEFORE building obs: the observation returned for
        # this step must contain the action just applied (a_t), not a_{t-1}.
        # (Reward's action-rate term above already used the old value.)
        self._prev_action = action.copy()
        obs = self._get_obs()

        # --- Termination ---
        height = self.data.xpos[self._base_id][2]
        fell = height < self.min_height
        tilted = grav[2] > self.max_tilt_gz
        bad_state = bool(np.any(np.isnan(self.data.qpos)))
        terminated = fell or tilted or bad_state
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "vx": float(vx),
            "height": float(height),
            "r_forward": float(r_forward),
            "r_upright": float(r_upright),
            "r_action_rate": float(r_action_rate),
            "r_torque": float(r_torque),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
