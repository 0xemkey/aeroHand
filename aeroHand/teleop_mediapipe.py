"""Webcam teleoperation of the Aero Hand in MuJoCo using Google MediaPipe hand landmarks.

MediaPipe's hand landmarker is already trained, so no training is needed here. A quick per-user
calibration (open hand / fist) maps your 21 hand landmarks onto the robot's 7 finger actuators.
Your hand's position and orientation drive the robot wrist (6 DoF) while wrist tracking is on.

Keys (focus the camera window):
  o      hold your hand OPEN (fingers straight, thumb out) and press -> calibrate open pose
  c      make a FIST and press                                       -> calibrate closed pose
  w      toggle wrist tracking (on at start; works like a clutch): turning it on takes your current
         hand pose as the reference, so turn it off, move your real hand back to a comfortable
         spot, turn it on again
  (run with --no-move to let the camera rotate the wrist but not move it)
  i/k j/l u/n   keyboard wrist move: forward/back, left/right, up/down (1 cm per press)
  1/2 3/4 5/6   keyboard wrist rotate: yaw, pitch, roll (5 deg per press)
  (the tool starts softly attached to the palm and follows the hand; close your hand fully and
   it is handed over to the fingers automatically)
  g      re-attach the tool to the palm (e.g. after dropping it)
  space  release the tool from the palm now
  r      reset the simulation
  q/Esc  quit
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import mujoco
import mujoco.viewer
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

ROOT = Path(__file__).parent
DEFAULT_SCENE = ROOT / "aero-open-sim" / "mujoco" / "scene_laparoscope.xml"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CALIB_PATH = ROOT / "models" / "hand_calibration.json"

# Feature order matches the finger actuator order in right_hand.xml:
# index, middle, ring, pinky, thumb_abd, th1 (thumb swing across palm), th2 (thumb curl)
FEATURES = ["index", "middle", "ring", "pinky", "thumb_abd", "thumb_swing", "thumb_curl"]
NU_HAND = len(FEATURES)
# Rough defaults (open, fist) used until you calibrate.
DEFAULT_CALIB = {
    "open": [0.3, 0.3, 0.3, 0.3, 0.2, 1.4, 0.2],
    "fist": [2.8, 2.8, 2.8, 2.8, 0.9, 0.6, 1.3],
}
# Robot thumb rotation for an open hand / a fist (rad); 0 = flat beside palm, 1.745 = opposed.
# Capped at 0.6: in the laparoscope scene a more opposed thumb pushes the handle out of the grip.
# Raise toward 1.745 for small objects that need a pinch.
ROBOT_ABD_OPEN, ROBOT_ABD_FIST = 0.2, 0.6
FINGER_LANDMARKS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
SMOOTHING = 0.4  # EMA weight of the newest measurement
CALIB_FRAMES = 15
# The tool starts softly attached to the palm; it is handed over to the fingers once the mean finger
# closure stays >= GRIP_RELEASE for GRIP_HOLD_TIME s (in sim tests >= 0.85 held in any wrist pose).
GRIP_RELEASE, GRIP_HOLD_TIME = 0.85, 0.5

# --- Wrist mapping -------------------------------------------------------------------------------
WRIST_ACTUATORS = ["wrist_x_A", "wrist_y_A", "wrist_z_A", "wrist_yaw_A", "wrist_pitch_A", "wrist_roll_A"]
# Robot meters per full image width/height of hand motion, and for depth (from apparent hand size).
POS_GAIN_IMAGE, POS_GAIN_DEPTH = 0.5, 0.4
ROT_GAIN = 1.0
# Camera axes (x right, y down, z away from camera; image is mirrored) -> robot world axes
# (x = fingers forward, y = left, z = up). Moving your hand toward the camera moves the robot
# forward, image-right moves it to robot -y, image-up moves it up. Flip a sign here if an axis
# feels inverted.
CAM_TO_WORLD = np.array([[0.0, 0.0, -1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
KEY_STEP_POS, KEY_STEP_ROT = 0.01, np.radians(5)
WRIST_KEYS = {ord("i"): (0, +KEY_STEP_POS), ord("k"): (0, -KEY_STEP_POS),
              ord("j"): (1, +KEY_STEP_POS), ord("l"): (1, -KEY_STEP_POS),
              ord("u"): (2, +KEY_STEP_POS), ord("n"): (2, -KEY_STEP_POS),
              ord("1"): (3, +KEY_STEP_ROT), ord("2"): (3, -KEY_STEP_ROT),
              ord("3"): (4, +KEY_STEP_ROT), ord("4"): (4, -KEY_STEP_ROT),
              ord("5"): (5, +KEY_STEP_ROT), ord("6"): (5, -KEY_STEP_ROT)}


def angle(a, b):
    """Unsigned angle between two vectors (rad)."""
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def hand_features(p):
    """p: (21, 3) MediaPipe world landmarks (meters). Returns the 7 raw finger features."""
    f = []
    for mcp, pip, dip, tip in FINGER_LANDMARKS.values():
        f.append(angle(p[mcp] - p[0], p[pip] - p[mcp])
                 + angle(p[pip] - p[mcp], p[dip] - p[pip])
                 + angle(p[dip] - p[pip], p[tip] - p[dip]))
    palm_normal = np.cross(p[5] - p[0], p[17] - p[0])
    palm_normal /= np.linalg.norm(palm_normal) + 1e-9
    thumb_dir = p[3] - p[1]
    # How far the thumb points out of the palm plane (opposition).
    f.append(abs(np.pi / 2 - angle(thumb_dir, palm_normal)))
    # Thumb tip to pinky base, normalised by palm length: small when the thumb crosses the palm.
    f.append(float(np.linalg.norm(p[4] - p[17]) / (np.linalg.norm(p[9] - p[0]) + 1e-9)))
    f.append(angle(p[2] - p[1], p[3] - p[2]) + angle(p[3] - p[2], p[4] - p[3]))
    return np.array(f)


def palm_rotation(p):
    """Palm frame (columns: across palm toward index, toward fingers, palm normal) in camera axes."""
    ey = p[9] - p[0]
    ey /= np.linalg.norm(ey) + 1e-9
    ex = p[5] - p[17]
    ex -= ey * np.dot(ex, ey)
    ex /= np.linalg.norm(ex) + 1e-9
    return np.column_stack([ex, ey, np.cross(ex, ey)])


def palm_image_pose(lm):
    """Palm center (u, v) in normalized image coords and apparent palm size."""
    pts = np.array([[lm[i].x, lm[i].y] for i in (0, 5, 9, 13, 17)])
    size = np.mean([np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[0] - pts[4]), np.linalg.norm(pts[1] - pts[4])])
    return pts.mean(axis=0), size


def rot_to_zyx(R):
    """Rotation matrix -> (yaw, pitch, roll) with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return np.array([np.arctan2(R[1, 0], R[0, 0]), np.arcsin(np.clip(-R[2, 0], -1, 1)), np.arctan2(R[2, 1], R[2, 2])])


def zyx_to_rot(yaw, pitch, roll):
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def scale_rotation(R, gain):
    """Scale the angle of rotation R by gain (axis-angle)."""
    if gain == 1.0:
        return R
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    axis, ang = np.zeros(3), 0.0
    if np.linalg.norm(q[1:]) > 1e-9:
        ang = 2 * np.arctan2(np.linalg.norm(q[1:]), q[0])
        axis = q[1:] / np.linalg.norm(q[1:])
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, axis, ang * gain)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, out)
    return m.reshape(3, 3)


def load_calibration():
    if CALIB_PATH.exists():
        return json.loads(CALIB_PATH.read_text())
    return {k: list(v) for k, v in DEFAULT_CALIB.items()}


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 3, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--camera", default="0",
                        help="webcam index (0, 1, ...) or a stream URL, e.g. http://192.168.1.20:8080/video")
    parser.add_argument("--video", help="use a video file instead of the webcam")
    parser.add_argument("--list-cameras", action="store_true", help="print which camera indices work and exit")
    parser.add_argument("--no-move", action="store_true",
                        help="camera drives wrist rotation only; position stays put (keyboard i/k j/l u/n still move it)")
    args = parser.parse_args()

    if args.list_cameras:
        for i in range(6):
            cap = cv2.VideoCapture(i)
            ok = cap.isOpened() and cap.read()[0]
            print(f"camera {i}: {'OK' if ok else '-'}")
            cap.release()
        return
    camera = int(args.camera) if args.camera.isdigit() else args.camera

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    has_instrument = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "laparoscope_free") >= 0
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pregrasp")
    wrist_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in WRIST_ACTUATORS]
    has_wrist = all(i >= 0 for i in wrist_ids)
    if has_wrist:
        wrist_lo, wrist_hi = model.actuator_ctrlrange[wrist_ids].T

    lo, hi = model.actuator_ctrlrange[:NU_HAND, 0], model.actuator_ctrlrange[:NU_HAND, 1]
    ctrl_open, ctrl_fist = hi.copy(), lo.copy()
    ctrl_open[4], ctrl_fist[4] = ROBOT_ABD_OPEN, ROBOT_ABD_FIST

    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "tool_in_palm")
    has_weld = has_instrument and weld_id >= 0
    if has_instrument:
        qadr = model.jnt_qposadr[model.joint("laparoscope_free").id]
        vadr = model.jnt_dofadr[model.joint("laparoscope_free").id]
        tool_body = model.body("laparoscope").id
        palm_body = model.body("palm").id

    state = {"attached": has_weld, "grip_time": 0.0, "wrist": np.zeros(6), "tracking": has_wrist, "ref": None}

    def attach():
        """Put the tool back in the palm (pose from the weld's relpose) and hold it there softly."""
        mujoco.mj_kinematics(model, data)
        rel_pos, rel_quat = model.eq_data[weld_id, 3:6], model.eq_data[weld_id, 6:10]
        data.qpos[qadr:qadr + 3] = data.xpos[palm_body] + data.xmat[palm_body].reshape(3, 3) @ rel_pos
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, data.xquat[palm_body], rel_quat)
        data.qpos[qadr + 3:qadr + 7] = quat
        data.qvel[vadr:vadr + 6] = 0
        data.eq_active[weld_id] = 1
        state.update(attached=True, grip_time=0.0)

    def release():
        data.eq_active[weld_id] = 0
        state["attached"] = False

    def reset():
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        state.update(wrist=np.zeros(6), tracking=has_wrist, ref=None)
        if has_weld:
            attach()
        mujoco.mj_forward(model, data)

    reset()

    calib = load_calibration()
    blend = np.zeros(NU_HAND)  # 0 = open, 1 = fist, per finger actuator
    collecting, samples = None, []

    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    ))
    cap = cv2.VideoCapture(args.video if args.video else camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video. Try --list-cameras, --camera 1 or --video file.mp4")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low for network (phone) streams

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last = time.time()
        while viewer.is_running():
            ok, frame = cap.read()
            if not ok:
                break
            if not args.video:
                frame = cv2.flip(frame, 1)  # mirror so it feels like a mirror
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time.time() * 1000))

            if result.hand_world_landmarks:
                pts = np.array([[l.x, l.y, l.z] for l in result.hand_world_landmarks[0]])
                lms = result.hand_landmarks[0]
                feats = hand_features(pts)
                draw_landmarks(frame, lms)
                if collecting:
                    samples.append(feats)
                    if len(samples) >= CALIB_FRAMES:
                        calib[collecting] = np.mean(samples, axis=0).tolist()
                        CALIB_PATH.write_text(json.dumps(calib, indent=2))
                        print(f"Calibrated '{collecting}': {np.round(calib[collecting], 2)}")
                        collecting, samples = None, []
                f_open, f_fist = np.array(calib["open"]), np.array(calib["fist"])
                span = np.where(np.abs(f_fist - f_open) < 1e-6, 1e-6, f_fist - f_open)
                target = np.clip((feats - f_open) / span, 0.0, 1.0)
                blend = (1 - SMOOTHING) * blend + SMOOTHING * target

                if has_wrist and state["tracking"]:
                    (u, v), size = palm_image_pose(lms)
                    R_cam = palm_rotation(pts)
                    if state["ref"] is None:  # clutch engaged: remember where everything started
                        state["ref"] = dict(uv=np.array([u, v]), size=size, R=R_cam, wrist=state["wrist"].copy())
                    ref = state["ref"]
                    du, dv = np.array([u, v]) - ref["uv"]
                    d_depth = ref["size"] / size - 1.0  # > 0 when the hand moves away from the camera
                    d_cam = np.array([du * POS_GAIN_IMAGE, dv * POS_GAIN_IMAGE, d_depth * POS_GAIN_DEPTH])
                    pos = ref["wrist"][:3] + (0 if args.no_move else CAM_TO_WORLD @ d_cam)
                    R_rel = scale_rotation(CAM_TO_WORLD @ (R_cam @ ref["R"].T) @ CAM_TO_WORLD.T, ROT_GAIN)
                    rpy = rot_to_zyx(R_rel @ zyx_to_rot(*ref["wrist"][3:]))
                    goal = np.clip(np.concatenate([pos, rpy]), wrist_lo, wrist_hi)
                    state["wrist"] = (1 - SMOOTHING) * state["wrist"] + SMOOTHING * goal

            data.ctrl[:NU_HAND] = ctrl_open + blend * (ctrl_fist - ctrl_open)
            if has_wrist:
                data.ctrl[wrist_ids] = state["wrist"]

            # Advance the simulation by the wall-clock time since the last frame.
            now = time.time()
            n_steps = min(int((now - last) / model.opt.timestep), 20)
            last += n_steps * model.opt.timestep
            with viewer.lock():
                # Auto-grab: once your hand has been nearly closed for a moment, drop the soft
                # attachment so only the fingers hold the tool.
                if state["attached"]:
                    closed = np.mean(blend[:4]) >= GRIP_RELEASE
                    state["grip_time"] = state["grip_time"] + n_steps * model.opt.timestep if closed else 0.0
                    if state["grip_time"] >= GRIP_HOLD_TIME:
                        release()
                        print("Tool grabbed - now held by the fingers only (open your hand to drop it, g to re-attach)")
                for _ in range(n_steps):
                    mujoco.mj_step(model, data)
            viewer.sync()
            n_contacts = sum(1 for c in data.contact[:data.ncon]
                             if (model.geom_bodyid[c.geom1] == tool_body) != (model.geom_bodyid[c.geom2] == tool_body)
                             ) if has_instrument else 0

            hud = " ".join(f"{n[:5]}:{b:.2f}" for n, b in zip(FEATURES, blend))
            cv2.putText(frame, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if has_wrist:
                w = state["wrist"]
                mode = "TRACKING rotation only" if args.no_move else "TRACKING"
                wrist_txt = (f"wrist {mode if state['tracking'] else 'off (w)'}  "
                             f"xyz {w[0]:+.2f} {w[1]:+.2f} {w[2]:+.2f} m  "
                             f"ypr {np.degrees(w[3]):+.0f} {np.degrees(w[4]):+.0f} {np.degrees(w[5]):+.0f} deg")
                cv2.putText(frame, wrist_txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            if collecting:
                status = "CALIBRATING " + collecting
            elif not has_instrument:
                status = ""
            elif state["attached"]:
                status = f"tool in palm - CLOSE your hand to grab ({n_contacts} contacts)"
            else:
                status = f"tool free ({n_contacts} contacts) - g re-attach"
            cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, "o/c calibrate  w wrist  g re-attach tool  r reset  q quit", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, "Keys only work while THIS window is focused (click it)", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            if not result.hand_world_landmarks:
                cv2.putText(frame, "NO HAND DETECTED", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Aero Hand teleop (MediaPipe)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                print(f"key '{chr(key)}' pressed", end="")
                if key == ord("w") and has_wrist:
                    print(f" -> wrist tracking {'OFF' if state['tracking'] else 'ON'}", end="")
                elif key in WRIST_KEYS and has_wrist:
                    print(f" -> wrist {WRIST_ACTUATORS[WRIST_KEYS[key][0]][:-2]} {WRIST_KEYS[key][1]:+.3f}", end="")
                print()
            if key in (ord("q"), 27):
                break
            elif key == ord("o"):
                collecting, samples = "open", []
            elif key == ord("c"):
                collecting, samples = "fist", []
            elif key == ord(" ") and has_weld:
                release()
            elif key == ord("g") and has_weld:
                with viewer.lock():
                    attach()
            elif key == ord("r"):
                with viewer.lock():
                    reset()
            elif key == ord("w") and has_wrist:
                state["tracking"] = not state["tracking"]
                state["ref"] = None
            elif key in WRIST_KEYS and has_wrist:
                idx, step = WRIST_KEYS[key]
                state["wrist"][idx] = np.clip(state["wrist"][idx] + step, wrist_lo[idx], wrist_hi[idx])
                state["ref"] = None  # re-anchor camera tracking at the new pose

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
