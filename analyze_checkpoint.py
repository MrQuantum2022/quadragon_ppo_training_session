"""
Quantitative gait analysis for a trained checkpoint - goes beyond watching
video by actually measuring foot contact patterns, joint trajectories, and
leg coordination directly from MuJoCo state.

Usage:
    python3 analyze_checkpoint.py <model.zip> <vecnorm.pkl> --version v2 --save-plot v2_analysis.png

Works for v1 or v2 checkpoints - foot contact/height are read directly from
the shared physics model (quadragon_mg995.xml), independent of whether the
env class you're analyzing exposes foot state in its own observation.

Produces:
    - A saved PNG with 4 panels: gait diagram, forward velocity, body
      height/tilt, and calf joint angle trajectories
    - Printed summary: distance, duty factor per foot, stride frequency per
      foot, and diagonal/lateral coordination scores (checks for emerging
      trot-like coordination even before any phase-timing reward exists)
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive - we only save PNGs, never show a window.
                        # Avoids matplotlib auto-detecting Qt/GTK/Tk, which can
                        # crash silently (no traceback) on machines with a
                        # broken/partial GUI toolkit install - exactly what we
                        # already saw with MuJoCo's GLFW renderer on this box.
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def get_env_class(version: str):
    if version == "v1":
        from quadragon_env import QuadragonEnv
        return QuadragonEnv
    elif version == "v2":
        from quadragon_env_v2 import QuadragonEnvV2
        return QuadragonEnvV2
    elif version == "v3a":
        from quadragon_env_v3a import QuadragonEnvV3a
        return QuadragonEnvV3a
    elif version == "v3b":
        from quadragon_env_v3b import QuadragonEnvV3b
        return QuadragonEnvV3b
    elif version == "v4":
        from quadragon_env_v4 import QuadragonEnvV4
        return QuadragonEnvV4
    elif version == "v5":
        from quadragon_env_v5 import QuadragonEnvV5
        return QuadragonEnvV5
    else:
        from quadragon_env_v6 import QuadragonEnvV6
        return QuadragonEnvV6


def foot_sensors(raw_env):
    """Foot geoms exist in the shared MJCF regardless of env version -
    read them directly rather than depending on what the env's own obs exposes."""
    foot_names = ["FR_foot", "BR_foot", "BL_foot", "FL_foot"]
    foot_gids = [raw_env.model.geom(n).id for n in foot_names]
    floor_gid = raw_env.model.geom("floor").id

    def contacts():
        data = raw_env.data
        flags = {int(g): False for g in foot_gids}
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if floor_gid in pair:
                for g in foot_gids:
                    if int(g) in pair:
                        flags[int(g)] = True
        return np.array([flags[int(g)] for g in foot_gids])

    def heights():
        return np.array([raw_env.data.geom_xpos[g][2] for g in foot_gids])

    return contacts, heights


def rollout_and_record(model, venv, raw_env, n_steps: int):
    get_contacts, get_heights = foot_sensors(raw_env)

    obs = venv.reset()
    log = {k: [] for k in ["t", "qpos", "vx", "height", "roll", "pitch",
                            "foot_contact", "foot_height"]}

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = venv.step(action)

        grav = raw_env._projected_gravity()
        roll = np.arctan2(grav[1], -grav[2])
        pitch = np.arctan2(-grav[0], -grav[2])

        log["t"].append(step * raw_env.dt)
        log["qpos"].append(raw_env.data.qpos[raw_env._act_qpos_adr].copy())
        log["vx"].append(info[0]["vx"])
        log["height"].append(info[0]["height"])
        log["roll"].append(roll)
        log["pitch"].append(pitch)
        log["foot_contact"].append(get_contacts())
        log["foot_height"].append(get_heights())

        if done[0]:
            break

    return {k: np.array(v) for k, v in log.items()}


def analyze(log):
    t = log["t"]
    duration = t[-1] - t[0] if len(t) > 1 else 1e-9
    dt = t[1] - t[0] if len(t) > 1 else 0.02
    contacts = log["foot_contact"]
    foot_names = ["FR", "BR", "BL", "FL"]

    print(f"Episode length: {len(t)} steps ({duration:.2f}s)")
    print(f"Mean forward velocity: {np.mean(log['vx']):.3f} m/s")
    print(f"Distance traveled (integrated vx): {np.sum(log['vx']) * dt:.3f} m")
    print(f"Mean body height: {np.mean(log['height']):.3f} m (std {np.std(log['height']):.4f})")
    print(f"Roll std: {np.degrees(np.std(log['roll'])):.2f} deg   "
          f"Pitch std: {np.degrees(np.std(log['pitch'])):.2f} deg")
    print()

    # --- Per-joint usage: turns "that joint looks frozen" into numbers ---
    joint_names = ["FR_hip", "FR_thigh", "FR_calf", "BR_hip", "BR_thigh", "BR_calf",
                   "BL_hip", "BL_thigh", "BL_calf", "FL_hip", "FL_thigh", "FL_calf"]
    # available range per joint type (deg): hip +/-40, thigh +/-60, calf -90..+10
    avail = {"hip": 80.0, "thigh": 120.0, "calf": 100.0}
    qdeg = np.degrees(log["qpos"])
    print("Per-joint usage (deg): mean / std / span, and span as % of available range")
    print("(a joint sitting still shows tiny std and span %)")
    for i, jn in enumerate(joint_names):
        kind = jn.split("_")[1]
        span = float(qdeg[:, i].max() - qdeg[:, i].min())
        pct = 100.0 * span / avail[kind]
        print(f"  {jn:9s} mean {qdeg[:, i].mean():+7.1f}  std {qdeg[:, i].std():5.1f}  "
              f"span {span:5.1f}  ({pct:4.0f}% of range)")
    print()

    print("Per-foot duty factor (fraction of episode in ground contact):")
    for name, d in zip(foot_names, contacts.mean(axis=0)):
        print(f"  {name}: {d:.2f}")
    print()

    print("Stride count / frequency (stance->swing transitions):")
    for i, name in enumerate(foot_names):
        transitions = int(np.sum(np.diff(contacts[:, i].astype(int)) == -1))
        freq = transitions / duration
        print(f"  {name}: {transitions} steps, ~{freq:.2f} Hz")
    print()

    fr, br, bl, fl = (contacts[:, i] for i in range(4))

    def phase_correlation(a, b):
        """Pearson correlation between two feet's contact series. Unlike raw
        state-matching, this correctly nets out each foot's own duty-factor
        bias - two feet that are each independently ~90% swing will score
        near 0 here even though they'd 'match' most of the time by chance."""
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(np.corrcoef(a.astype(float), b.astype(float))[0, 1])

    print("Coordination (phase correlation, -1 to +1, 0 = no real relationship")
    print("regardless of duty factor - NOT raw state-matching):")
    print(f"  Diagonal FR/BL: {phase_correlation(fr, bl):+.2f}")
    print(f"  Diagonal FL/BR: {phase_correlation(fl, br):+.2f}")
    print(f"  Front L/R:      {phase_correlation(fr, fl):+.2f}")
    print("  A real trot: diagonal pairs correlate POSITIVELY (move together);")
    print("  the two diagonal pairs correlate NEGATIVELY against each other")
    print("  (one pair planted while the other swings). Near 0 anywhere =")
    print("  no real timing relationship there yet, whatever the duty factors say.")


def plot_analysis(log, save_path):
    t = log["t"]
    foot_names = ["FR", "BR", "BL", "FL"]
    contacts = log["foot_contact"]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1, 1, 1.6]})

    ax = axes[0]
    for i, name in enumerate(foot_names):
        stance = contacts[:, i].astype(int)
        changes = np.diff(np.concatenate([[0], stance, [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        for s, e in zip(starts, ends):
            e_idx = min(e, len(t) - 1)
            ax.barh(i, t[e_idx] - t[s], left=t[s], height=0.6, color="#2a72d4")
    ax.set_yticks(range(4))
    ax.set_yticklabels(foot_names)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("Gait diagram (filled = foot in ground contact)")

    ax = axes[1]
    ax.plot(t, log["vx"], color="#2a72d4")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("vx (m/s)")
    ax.set_title("Forward velocity")

    ax = axes[2]
    ax.plot(t, log["height"], color="#2a72d4")
    ax.set_ylabel("height (m)", color="#2a72d4")
    ax2 = ax.twinx()
    ax2.plot(t, np.degrees(log["roll"]), color="#c1503f", alpha=0.6, label="roll")
    ax2.plot(t, np.degrees(log["pitch"]), color="#e8934a", alpha=0.6, label="pitch")
    ax2.set_ylabel("deg")
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_title("Body height & orientation")

    ax = axes[3]
    joint_names = ["FR_calf", "BR_calf", "BL_calf", "FL_calf"]
    calf_idx = [2, 5, 8, 11]
    qpos = log["qpos"]
    for i, jn in zip(calf_idx, joint_names):
        ax.plot(t, np.degrees(qpos[:, i]), label=jn, alpha=0.85)
    ax.set_ylabel("calf angle (deg)")
    ax.set_xlabel("time (s)")
    ax.set_title("Calf joint trajectories (the joints the 42deg decode fix touched)")
    ax.legend(fontsize=8, ncol=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    print(f"\nSaved plot to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("vecnorm_path")
    p.add_argument("--version", choices=["v1", "v2", "v3a", "v3b", "v4", "v5", "v6"], default="v2")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--save-plot", default="checkpoint_analysis.png")
    args = p.parse_args()

    EnvClass = get_env_class(args.version)
    venv = DummyVecEnv([lambda: EnvClass()])
    venv = VecNormalize.load(args.vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(args.model_path)
    raw_env = venv.venv.envs[0]

    print(f"Forward axis: {getattr(raw_env, 'forward_axis', '+x (legacy)')}\n")
    log = rollout_and_record(model, venv, raw_env, args.steps)
    analyze(log)
    plot_analysis(log, args.save_plot)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        print("\n--- analyze_checkpoint.py crashed - full traceback below ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
