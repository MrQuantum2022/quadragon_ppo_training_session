"""
Quadragon v4 - random push perturbations (dynamic-balance pressure).

Subclasses QuadragonEnvV2 directly - NOT v3a/v3b. The v3 A/B showed both
coordination-reward arms regressed locomotion (~15-20% slower than v2) while
moving coordination correlations not at all (still inside the +/-0.16 noise
band). Root cause of the frozen-diagonal "leaning gait": static two-leg
balance is perfectly viable in a world with no disturbances, so no reward
shaping dislodges it. v4 attacks the cause instead of the symptom: random
horizontal shoves applied to the body during training. A robot parked on one
diagonal gets knocked over (existing fall-termination already punishes that);
a robot that genuinely steps can recover. Standard push-robustness training.

Implementation: a random horizontal force applied to base_link via MuJoCo's
xfrc_applied for a short burst, at random intervals. No spawned objects, no
collision machinery - identical training pressure, ~15 lines.

Obs: UNCHANGED from v2 (50 dims). The policy senses pushes the same way the
real robot would - through their effect on IMU/velocity - so nothing here
breaks sim-to-real.

Force calibration (EMPIRICAL - the naive free-body delta-v = F*t/m estimate
is wrong here, because a planted robot absorbs horizontal force through foot
friction + actuated stance; ~12 N of weight on the feet soaks up small pushes
entirely). Measured against a passive 4-foot stance, the robot's MOST stable
configuration: 5-7 N -> negligible (0.8 deg tilt); 10-14 N -> real disturbance
(8.6 deg) but survivable; 15-20 N -> topples even the passive 4-foot stance.
Defaults (8-16 N, 0.2 s) span "clearly felt" to "borderline for a solid
stance" - guaranteed lethal to a static two-leg lean, recoverable by a robot
that actually steps. If training stalls with ep_len collapsed (pushes too
brutal to learn under), lower the top of push_force_range to ~12 N.

All randomness draws from self.np_random -> deterministic under a fixed seed,
consistent with every other env in this project.
"""

from __future__ import annotations

import numpy as np

from quadragon_env_v2 import QuadragonEnvV2


class QuadragonEnvV4(QuadragonEnvV2):
    def __init__(self,
                 push_force_range: tuple = (8.0, 16.0),    # Newtons, horizontal (empirically calibrated)
                 push_duration_s: float = 0.2,             # how long each shove lasts
                 push_interval_range_s: tuple = (1.0, 3.0),  # random gap between shoves
                 push_grace_period_s: float = 1.0,         # no pushes right after reset
                 **kwargs):
        super().__init__(**kwargs)

        self.push_force_range = push_force_range
        self.push_duration_steps = max(1, int(push_duration_s / self.dt))
        self.push_interval_range_s = push_interval_range_s
        self.push_grace_steps = int(push_grace_period_s / self.dt)

        self._push_steps_left = 0
        self._push_force = np.zeros(3)
        self._next_push_at = 0
        self._push_count = 0

    # ---- internals ----

    def _schedule_next_push(self):
        gap_s = self.np_random.uniform(*self.push_interval_range_s)
        self._next_push_at = self._step_count + max(1, int(gap_s / self.dt))

    def _start_push(self):
        magnitude = self.np_random.uniform(*self.push_force_range)
        angle = self.np_random.uniform(0, 2 * np.pi)
        # Horizontal-only force in the world frame; no vertical component
        self._push_force = np.array([magnitude * np.cos(angle),
                                     magnitude * np.sin(angle),
                                     0.0])
        self._push_steps_left = self.push_duration_steps
        self._push_count += 1

    def _apply_or_clear_push(self):
        """Write the current push force (or zeros) into xfrc_applied for base_link."""
        self.data.xfrc_applied[self._base_id, :3] = (
            self._push_force if self._push_steps_left > 0 else 0.0
        )

    # ---- gym API ----

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Clear any force left over from a previous episode
        self.data.xfrc_applied[self._base_id, :] = 0.0
        self._push_steps_left = 0
        self._push_force = np.zeros(3)
        self._push_count = 0
        # First push lands some time after the grace period
        gap_s = self.np_random.uniform(*self.push_interval_range_s)
        self._next_push_at = self.push_grace_steps + max(1, int(gap_s / self.dt))
        return obs, info

    def step(self, action):
        # Start a new push if it's time and none is active
        if self._push_steps_left <= 0 and self._step_count >= self._next_push_at:
            self._start_push()
            self._schedule_next_push()

        self._apply_or_clear_push()

        obs, reward, terminated, truncated, info = super().step(action)

        if self._push_steps_left > 0:
            self._push_steps_left -= 1
            if self._push_steps_left == 0:
                # Push just ended - clear the force so it doesn't leak into
                # subsequent physics steps
                self.data.xfrc_applied[self._base_id, :] = 0.0

        info["push_active"] = bool(self._push_steps_left > 0)
        info["push_count"] = self._push_count
        return obs, reward, terminated, truncated, info
