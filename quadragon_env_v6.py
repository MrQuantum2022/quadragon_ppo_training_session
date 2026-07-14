"""
Quadragon v6 - corrected forward direction.

Subclasses QuadragonEnvV5 (keeps pushes + MG995 servo dynamics). The ONLY
change: forward is now along the body's TRUE long axis.

WHY: the leg-stance rectangle measures 0.101 m along X vs 0.151 m along Y -
the long (head-to-tail) axis is Y, but v1-v5 all trained forward = +X, the
SHORT axis: the robot was being taught to walk sideways, crab-style. Origin
of the error: the SolidWorks exporter's front_*/back_* link names sat at
+X/-X and were trusted over the physical body plan (Tushar, who built the
robot, caught it). Supporting evidence from v5's own behavior: BR_hip parked
at +21.7 deg mean sweeping 39% of range - the policy contorting its hips to
re-aim legs off-axis, consistent with the mechanics favoring the other axis.

SIGN AMBIGUITY: geometry says the long axis is Y, but only visual inspection
of the real robot says whether the head is at +Y or -Y. Default is '+y';
CONFIRM WITH THE ARROW in live_view before training (the v5 checkpoint is
obs-compatible - both 50-dim - so it can be loaded just to look):

    python3 live_view.py <v5_ckpt>.zip <v5_vecnorm>.pkl --version v6
    # arrow should point out of the robot's HEAD; if it points out the tail:
    python3 live_view.py ... --version v6 --forward -y

Then train with the confirmed sign (train.py --forward flag, default +y).

Prediction this version tests: after retraining along the true axis, hip
means should return near 0 deg (no more contortion). If the splay persists,
direction was NOT its cause and something else is driving it - either way
we learn something. Note the base COM sits offset along Y (SolidWorks origin
artifact places the body off-center between the legs on exactly this axis),
so some fore-aft asymmetry in the gait may be genuine and expected.

Obs unchanged (50): projected gravity, angular velocity, joint state are all
base-frame or joint-space quantities - none depend on which world axis is
called forward. Only the reward's velocity term and reported "vx" change.
"""

from __future__ import annotations

import numpy as np

from quadragon_env_v5 import QuadragonEnvV5

_AXES = {
    "+x": np.array([1.0, 0.0]),
    "-x": np.array([-1.0, 0.0]),
    "+y": np.array([0.0, 1.0]),
    "-y": np.array([0.0, -1.0]),
}


class QuadragonEnvV6(QuadragonEnvV5):
    def __init__(self, forward_axis: str = "+y", **kwargs):
        super().__init__(**kwargs)
        if forward_axis not in _AXES:
            raise ValueError(f"forward_axis must be one of {list(_AXES)}, got {forward_axis!r}")
        self.forward_axis = forward_axis
        self._forward_vec = _AXES[forward_axis]

    # _forward_velocity() is inherited - it already reads self._forward_vec.
    # Everything else (reward, termination, servo model, pushes) is untouched.
