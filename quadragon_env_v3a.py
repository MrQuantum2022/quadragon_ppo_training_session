"""
Quadragon v3a - data-driven (implicit) coordination.

Subclasses QuadragonEnvV2 (inherits decode fix + clearance + swing-cap).
The ONLY additions vs v2:
  1. Duty-factor balance reward: penalize the spread in how much each of the
     4 legs is used, pushing toward all legs sharing ground-contact work
     roughly equally (directly targets the FR/BL-underused imbalance).
  2. Stride-frequency penalty: penalize excessively rapid contact toggling
     (the ~8 Hz jitter seen on BR/FL), pushing toward slower deliberate steps.

NO phase signal in obs, NO prescribed rhythm. Coordination must EMERGE from
these pressures - this is the "let it emerge" arm of the v3 A/B experiment.

Obs: unchanged from v2 (50 dims). This arm deliberately adds no observation.

Reward shaping here uses a running window of recent contact history so duty
factor and toggle rate are meaningful per-step signals, not just end-of-episode
stats.
"""

from __future__ import annotations

import numpy as np

from quadragon_env_v2 import QuadragonEnvV2


class QuadragonEnvV3a(QuadragonEnvV2):
    def __init__(self, w_balance: float = 0.3, w_freq_penalty: float = 0.15,
                 window_s: float = 1.0, target_stride_hz: float = 3.0, **kwargs):
        super().__init__(**kwargs)

        self.w_balance = w_balance
        self.w_freq_penalty = w_freq_penalty

        # Rolling window of recent foot-contact states, for computing live
        # duty factor and toggle rate without waiting for episode end.
        self._window_len = max(1, int(window_s / self.dt))
        self._contact_history = None   # (window_len, 4), filled in reset

        # Above this per-foot stride frequency, the freq penalty kicks in.
        # Targets the ~8 Hz jitter while leaving a normal ~3 Hz step unpenalized.
        self.target_stride_hz = target_stride_hz
        self.window_s = window_s

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._contact_history = np.zeros((self._window_len, 4), dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        contact = self._feet_contact()

        # Roll the window: drop oldest, append current
        self._contact_history = np.roll(self._contact_history, -1, axis=0)
        self._contact_history[-1] = contact

        # --- Duty-factor balance: penalize spread across the 4 legs ---
        # Each leg's duty factor over the window; ideal = all equal, so we
        # penalize their standard deviation. Range of duty is [0,1] so std is
        # bounded and this term is well-scaled.
        duty = self._contact_history.mean(axis=0)   # (4,)
        balance_penalty = -self.w_balance * float(np.std(duty))

        # --- Stride-frequency penalty: count stance->swing transitions in the
        # window, convert to Hz, penalize only the excess above target ---
        transitions = np.sum(np.abs(np.diff(self._contact_history, axis=0)) > 0, axis=0)
        # each toggle counted; a full stride = 2 toggles (down->up->down), so /2
        stride_hz = (transitions / 2.0) / self.window_s   # (4,)
        excess = np.clip(stride_hz - self.target_stride_hz, 0, None)
        freq_penalty = -self.w_freq_penalty * float(np.mean(excess)) / self.target_stride_hz

        r_coord = balance_penalty + freq_penalty
        reward += r_coord

        info["r_balance"] = balance_penalty
        info["r_freq"] = freq_penalty
        info["duty_std"] = float(np.std(duty))
        return obs, reward, terminated, truncated, info
