"""Aero Hand Open MuJoCo demo.

  --scene hand         sweep all actuators open/closed
  --scene laparoscope  drop the laparoscopic instrument into the hand and grasp it
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MJCF_DIR = Path(__file__).parent / "aero-open-sim" / "mujoco"
SCENES = {"hand": MJCF_DIR / "scene_right.xml", "laparoscope": MJCF_DIR / "scene_laparoscope.xml"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=SCENES, default="hand")
    parser.add_argument("--headless", action="store_true", help="step without a viewer (sanity check)")
    parser.add_argument("--period", type=float, default=4.0, help="seconds per open/close cycle (hand scene)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENES[args.scene]))
    data = mujoco.MjData(model)

    print(f"nq={model.nq} nv={model.nv} nu={model.nu} ntendon={model.ntendon}")
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  actuator {i}: {model.actuator(i).name:28s} ctrlrange=[{lo:.4f}, {hi:.4f}]")

    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

    if args.scene == "laparoscope":
        mujoco.mj_resetDataKeyframe(model, data, model.key("pregrasp").id)
        pregrasp = data.ctrl.copy()
        grasp = pregrasp.copy()
        grasp[:7] = lo[:7]  # finger actuators: shortest tendons = fingers closed
        grasp[4] = 0.6      # thumb only partly over the palm; more (>~1.0) pushes the handle loose
        wrist = [model.actuator(n).id for n in ("wrist_z_A", "wrist_yaw_A", "wrist_roll_A")]

        def ramp(t, start, dur=1.5):
            return np.clip((t - start) / dur, 0.0, 1.0)

        def control(t):
            # settle into the cupped hand (1 s), close (1 s), lift 10 cm, roll 60 deg, yaw 45 deg, then
            # undo the motion; repeats every 14 s.
            t = t % 14.0
            c = pregrasp + np.clip(t - 1.0, 0.0, 1.0) * (grasp - pregrasp)
            back = ramp(t, 10.0, 2.5)
            c[wrist[0]] = 0.10 * ramp(t, 2.5) * (1 - back)
            c[wrist[2]] = np.radians(60) * ramp(t, 4.5) * (1 - back)
            c[wrist[1]] = np.radians(45) * ramp(t, 6.5) * (1 - back)
            return c
    else:
        def control(t):
            # Smoothly sweep every actuator across its full control range.
            s = 0.5 * (1 - np.cos(2 * np.pi * t / args.period))
            return lo + s * (hi - lo)

    if args.headless:
        for _ in range(int(2 * args.period / model.opt.timestep)):
            data.ctrl[:] = control(data.time)
            mujoco.mj_step(model, data)
        print(f"Simulated {data.time:.2f}s OK, qpos finite: {np.isfinite(data.qpos).all()}")
        if args.scene == "laparoscope":
            palm = data.body("palm")
            grip = palm.xmat.reshape(3, 3).T @ (data.site("laparoscope_grip_center").xpos - palm.xpos)
            print(f"Wrist qpos: {np.round(data.qpos[:6], 3)}")
            print(f"Grip center in palm frame: {np.round(grip, 3)} (about [0, -0.03, 0.065] when held)")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            data.ctrl[:] = control(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
