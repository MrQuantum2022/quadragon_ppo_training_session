"""
Quadragon v5 - realistic servo dynamics (anti-vibration-gait).

Subclasses QuadragonEnvV4 (keeps pushes - they broke the leaning gait and
recruited the thighs). The ONLY addition: an MG995 actuator model between the
policy's decoded targets and MuJoCo's position actuators.

WHY: every version v1-v4 converged to tiny rapid steps (7-10 Hz contact
toggling). Root cause is below the reward layer: MuJoCo's position actuators
track 50 Hz target changes essentially instantly, so high-frequency dithering
of small position targets produces foot vibration that friction converts into
net forward motion. It's the nearest optimum from a standing policy, so PPO
finds it every time, whatever the reward. A real MG995 (analog servo,
~0.16 s/60 deg slew, ~1 deg deadband) physically cannot execute that gait -
so this is both a sim exploit AND a sim-to-real trap.

FIX (servo realism, not reward shaping - unhackable by construction):
  1. SLEW LIMIT: the applied target can move at most `servo_slew_rad_s`
     (datasheet: 60 deg / 0.16 s at 6 V ~= 6.5 rad/s).
  2. FIRST-ORDER LAG: applied target relaxes toward the commanded target with
     time constant `filter_tau_s`. This is the dominant anti-dither term:
     at tau = 0.08 s (cutoff ~2 Hz), a 10 Hz dither is attenuated ~5x
     regardless of amplitude. Represents servo + linkage response; can be
     tuned against the real servo's measured step response later (the AS5600
     encoders on order are exactly the tool for that measurement).
  3. BACKLASH HYSTERESIS: the applied target only follows the lagged command
     once the error exceeds `backlash_rad` (~1.5 deg = analog-servo deadband
     ~0.9 deg + metal-gear lash). Crucially this is hysteresis on the ERROR,
     not a gate on command changes: every direction reversal eats 2x backlash
     of dead travel, which is the mechanism that actually kills small
     oscillations on real hardware. (First implementation gated command
     deltas instead - measured to barely attenuate dither; this replaces it.)

Obs unchanged (50). The policy still OUTPUTS at 50 Hz; the plant just responds
like the real one. Expected effect: contact-toggle frequency drops toward the
filter passband (~2-4 Hz); velocity may dip initially, then must be re-earned
with genuinely larger strides - the only mechanism left.
"""

from __future__ import annotations

import numpy as np

from quadragon_env_v4 import QuadragonEnvV4


class QuadragonEnvV5(QuadragonEnvV4):
    def __init__(self,
                 servo_slew_rad_s: float = 6.5,   # MG995 @ 6V: 60deg/0.16s
                 filter_tau_s: float = 0.12,      # small-signal lag time constant
                 backlash_rad: float = 0.026,     # ~1.5 deg deadband + gear lash
                 **kwargs):
        super().__init__(**kwargs)
        self.servo_slew_rad_s = servo_slew_rad_s
        self.filter_tau_s = filter_tau_s
        self.backlash_rad = backlash_rad
        # EMA coefficient for a first-order lag discretized at the control dt
        self._alpha = 1.0 - np.exp(-self.dt / max(filter_tau_s, 1e-6))
        self._applied_target = None   # what the "servo horn" is actually told

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Servo state starts at the actual settled joint positions
        self._applied_target = self.data.qpos[self._act_qpos_adr].copy()
        self._lagged = self._applied_target.copy()
        self.data.ctrl[:] = self._applied_target
        return obs, info

    def _servo_model(self, commanded: np.ndarray) -> np.ndarray:
        """Commanded target -> what the servo actually tracks this step."""
        prev = self._applied_target

        # 2. First-order lag: internal command state relaxes toward the input
        self._lagged = self._lagged + self._alpha * (commanded - self._lagged)

        # 3. Backlash hysteresis: output follows the lagged command only once
        # the error exceeds the backlash width, and only by the excess.
        err = self._lagged - prev
        follow = np.sign(err) * np.maximum(np.abs(err) - self.backlash_rad, 0.0)

        # 1. Slew limit on the output's motion per control step
        max_step = self.servo_slew_rad_s * self.dt
        step = np.clip(follow, -max_step, max_step)

        self._applied_target = prev + step
        return self._applied_target

    def step(self, action):
        # Intercept the parent's ctrl write: decode ourselves, pass through the
        # servo model, then let the parent chain run with ctrl pre-set. Parent
        # classes write data.ctrl from the raw decode - so we override by
        # decoding here and calling the grandparent chain with our filtered
        # target already applied via a small shim: we temporarily wrap
        # _decode_action to return the filtered target.
        commanded = self._decode_action(np.asarray(action, dtype=np.float32))
        filtered = self._servo_model(commanded)

        orig_decode = self._decode_action
        self._decode_action = lambda a: filtered
        try:
            obs, reward, terminated, truncated, info = super().step(action)
        finally:
            self._decode_action = orig_decode

        info["servo_lag"] = float(np.mean(np.abs(commanded - filtered)))
        return obs, reward, terminated, truncated, info
