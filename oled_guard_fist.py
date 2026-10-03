import argparse
import atexit
import ctypes
import statistics
import time

import cv2
import mediapipe as mp

HWND_BROADCAST = 0xFFFF
WM_SYSCOMMAND = 0x0112
SC_MONITORPOWER = 0xF170
MOUSEEVENTF_MOVE = 0x0001
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
kernel32.GetTickCount.restype = ctypes.c_uint


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def idle_seconds():
    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    user32.GetLastInputInfo(ctypes.byref(info))
    return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


class Screen:
    def __init__(self, mode):
        self.mode = mode
        self.is_on = True
        self.t_off = 0.0
        self.saved_brightness = 70
        if mode == "brightness":
            import screen_brightness_control as sbc
            self.sbc = sbc
            try:
                self.saved_brightness = sbc.get_brightness()[0]
            except Exception:
                pass

    def off(self):
        if not self.is_on:
            return
        self.is_on = False
        self.t_off = time.time()
        if self.mode == "brightness":
            try:
                self.saved_brightness = self.sbc.get_brightness()[0] or self.saved_brightness
            except Exception:
                pass
            self.sbc.set_brightness(0)
        else:
            user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)

    def on(self):
        if self.is_on:
            return
        self.is_on = True
        if self.mode == "brightness":
            self.sbc.set_brightness(self.saved_brightness)
        else:
            user32.PostMessageW(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, -1)
            user32.mouse_event(MOUSEEVENTF_MOVE, 1, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_MOVE, -1, 0, 0, 0)

    def check_external_wake(self):
        if self.is_on or self.mode != "monitor":
            return
        if idle_seconds() < (time.time() - self.t_off) - 0.3:
            self.is_on = True


def classify_hand(lm, thumb_limit, min_size):
    def dist(a, b):
        return ((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5

    size = dist(0, 9)
    thumb = dist(4, 17) / size if size > 1e-6 else 0.0

    if size < min_size:
        return "too small", size, thumb

    folded = sum(1 for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
                 if dist(0, tip) < dist(0, pip))
    if folded == 0:
        return "palm", size, thumb
    if folded == 4:
        if thumb > thumb_limit:
            return "thumb out", size, thumb
        return "fist", size, thumb
    return "other", size, thumb


def face_pose(lm):
    x1, x2 = lm[33].x, lm[263].x
    eye_y = (lm[33].y + lm[263].y) / 2
    d = lm[152].y - eye_y
    w = x2 - x1
    if d <= 1e-6 or abs(w) <= 1e-6:
        return None
    return (lm[1].y - eye_y) / d, (lm[1].x - x1) / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["monitor", "brightness"], default="monitor")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--wake", choices=["palm", "look", "both"], default="both")
    ap.add_argument("--fist-hold", type=float, default=0.4)
    ap.add_argument("--palm-hold", type=float, default=0.1)
    ap.add_argument("--palm-rearm", type=float, default=0.8)
    ap.add_argument("--rearm-delay", type=float, default=0.3)
    ap.add_argument("--thumb-limit", type=float, default=1.0)
    ap.add_argument("--min-hand-size", type=float, default=0.10)
    ap.add_argument("--pitch-threshold", type=float, default=0.07)
    ap.add_argument("--yaw-threshold", type=float, default=0.12)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--calib", type=float, default=3.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    screen = Screen(args.mode)
    atexit.register(screen.on)

    ES_CONTINUOUS = 0x80000000
    ES_DISPLAY_REQUIRED = 0x00000002
    kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED)
    atexit.register(lambda: kernel32.SetThreadExecutionState(ES_CONTINUOUS))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(
        max_num_hands=1, model_complexity=0,
        min_detection_confidence=0.6, min_tracking_confidence=0.5)

    print(f"Calibrating for {args.calib:.0f}s - look at the screen normally...")
    pitch_samples, yaw_samples = [], []
    base_pitch = base_yaw = None
    start = time.time()
    gap = 1.0 / args.fps

    fist_since = None
    away_since = None
    look_armed = False
    palm_since = None
    no_palm_since = None
    palm_armed = False

    def reset_wake_state():
        nonlocal away_since, look_armed, palm_since, no_palm_since, palm_armed
        away_since = None
        look_armed = False
        palm_since = None
        no_palm_since = None
        palm_armed = False

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.5)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()
            status = ""

            if base_pitch is None:
                res = face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    pose = face_pose(res.multi_face_landmarks[0].landmark)
                    if pose:
                        pitch_samples.append(pose[0])
                        yaw_samples.append(pose[1])
                if now - start >= args.calib:
                    if len(pitch_samples) < 5:
                        start = now
                        pitch_samples.clear()
                        yaw_samples.clear()
                    else:
                        base_pitch = statistics.median(pitch_samples)
                        base_yaw = statistics.median(yaw_samples)
                        print("Ready. Fist = screen off, open palm = screen on.")
                status = "calibrating..."

            elif screen.is_on:
                label, size, thumb = "no hand", 0.0, 0.0
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    label, size, thumb = classify_hand(
                        res.multi_hand_landmarks[0].landmark,
                        args.thumb_limit, args.min_hand_size)
                if label == "fist":
                    if fist_since is None:
                        fist_since = now
                    if now - fist_since >= args.fist_hold:
                        screen.off()
                        fist_since = None
                        reset_wake_state()
                else:
                    fist_since = None
                status = f"ON | {label} size={size:.2f} thumb={thumb:.2f}"

            else:
                screen.check_external_wake()
                label = "-"
                looking = False

                if args.wake in ("palm", "both"):
                    label = "no hand"
                    res = hands.process(rgb)
                    if res.multi_hand_landmarks:
                        label, _, _ = classify_hand(
                            res.multi_hand_landmarks[0].landmark,
                            args.thumb_limit, args.min_hand_size)
                    if label == "palm":
                        no_palm_since = None
                        if palm_armed:
                            if palm_since is None:
                                palm_since = now
                            if now - palm_since >= args.palm_hold:
                                screen.on()
                                reset_wake_state()
                    else:
                        palm_since = None
                        if no_palm_since is None:
                            no_palm_since = now
                        if now - no_palm_since >= args.palm_rearm:
                            palm_armed = True

                if args.wake in ("look", "both") and not screen.is_on:
                    res = face_mesh.process(rgb)
                    if res.multi_face_landmarks:
                        pose = face_pose(res.multi_face_landmarks[0].landmark)
                        if pose:
                            looking = ((pose[0] - base_pitch) < args.pitch_threshold and
                                       abs(pose[1] - base_yaw) < args.yaw_threshold)
                    if looking:
                        away_since = None
                        if look_armed:
                            screen.on()
                            reset_wake_state()
                    else:
                        if away_since is None:
                            away_since = now
                        if now - away_since >= args.rearm_delay:
                            look_armed = True

                status = (f"OFF | hand={label} palm_ready={palm_armed} "
                          f"looking={looking}")

            if args.debug:
                cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2)
                cv2.imshow("oled_guard_fist", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(max(0, gap - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        screen.on()


if __name__ == "__main__":
    main()