"""
Quadragon v3b - phase-clock (explicit) coordination.

Subclasses QuadragonEnvV2 (inherits decode fix + clearance + swing-cap).
Additions vs v2:
  1. A cyclic phase clock advancing at a fixed gait frequency. Its sin/cos are
     appended to the observation (2 dims) so the policy can perceive "where in
     the gait cycle am I".
  2. A target diagonal-trot contact pattern derived from that clock: diagonal
     pair A (FR+BL) should be in stance while diagonal pair B (FL+BR) swings,
     and vice versa, alternating each half-cycle. Reward matches actual foot
     contact to this target pattern.

This DIRECTLY prescribes a trot rhythm - the "prescribe it" arm of the v3 A/B
experiment. Very likely to produce clean diagonal coordination; the cost is the
reintroduced phase machinery (obs + target reward) that the clean rebuild
originally left behind.

Obs: v2's 50 + 2 (sin/cos phase) = 52 dims.

Foot index order (from v2): [FR, BR, BL, FL] = indices [0, 1, 2, 3].
Diagonal pair A = FR(0) + BL(2); diagonal pair B = BR(1) + FL(3).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from quadragon_env_v2 import QuadragonEnvV2


class QuadragonEnvV3b(QuadragonEnvV2):
    def __init__(self, gait_freq_hz: float = 2.0, w_phase: float = 0.4, **kwargs):
        super().__init__(**kwargs)

        self.gait_freq_hz = gait_freq_hz
        self.w_phase = w_phase
        self._phase = 0.0

        # Extend obs by 2 for sin/cos of the phase clock
        obs_dim = self.observation_space.shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # Diagonal pairs, in the [FR,BR,BL,FL] index convention
        self._diag_A = [0, 2]   # FR, BL
        self._diag_B = [1, 3]   # BR, FL

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._phase = 0.0
        return self._append_phase(obs), info

    def _append_phase(self, obs):
        return np.concatenate([
            obs, [np.sin(self._phase), np.cos(self._phase)]
        ]).astype(np.float32)

    def _target_stance(self):
        """Return the target contact pattern (1=stance, 0=swing) per foot for
        the current phase. First half of the cycle: diagonal A plants, B swings.
        Second half: swapped."""
        target = np.zeros(4, dtype=np.float32)
        # phase in [0, 2pi). sin(phase) >= 0 -> first half -> pair A stance.
        if np.sin(self._phase) >= 0:
            target[self._diag_A] = 1.0   # A stance, B swing
        else:
            target[self._diag_B] = 1.0   # B stance, A swing
        return target

    def step(self, action):
        # Advance the phase clock by one control step BEFORE scoring, so the
        # target pattern and the resulting contact are evaluated on the same tick.
        self._phase = (self._phase + 2 * np.pi * self.gait_freq_hz * self.dt) % (2 * np.pi)

        obs, reward, terminated, truncated, info = super().step(action)

        contact = self._feet_contact()
        target = self._target_stance()

        # Reward = fraction of feet whose actual contact matches the target
        # stance/swing state for this phase. 1.0 = perfect diagonal trot match.
        match = float(np.mean((contact > 0.5) == (target > 0.5)))
        r_phase = self.w_phase * match

        reward += r_phase
        info["r_phase"] = r_phase
        info["phase"] = float(self._phase)
        return self._append_phase(obs), reward, terminated, truncated, info
