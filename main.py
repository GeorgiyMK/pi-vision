# tracker.py
import argparse, json, os, time, math, threading
import numpy as np
import cv2
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ==== ПАРАМЕТРЫ АППАРАТУРЫ ====
MODEL_PATH = "yolov8n.pt"   # лёгкая модель; можно заменить на свой .pt
IMGSZ = 640                 # вход модели
CONF_THRES = 0.05
ROTATE_DEG = 180              # 0/90/180/270 если камера перевёрнута
SERIAL_BAUD = 115200
SERIAL_PORT_CANDIDATES = ["/dev/ttyACM0", "/dev/ttyUSB0"]
FRAME_W, FRAME_H = 1280, 720

# Диапазоны серво (подгоняются под механику)
PAN_MIN, PAN_MAX = 20, 160
TILT_MIN, TILT_MAX = 30, 150
PAN_HOME = (PAN_MIN + PAN_MAX) // 2
TILT_HOME = (TILT_MIN + TILT_MAX) // 2

# Точность удержания (в пикселях) для включения лазера
PIX_TOLERANCE = 10
# EMA сглаживание целевой точки (0..1, где 1 = без сглаживания)
EMA_ALPHA = 0.3
# Сканирование при потере цели
SCAN_SPEED_DEG = 0.6
SCAN_TILT_SWING = 10
# Таймауты
HEARTBEAT_PERIOD = 0.2
LOST_TARGET_TIMEOUT = 0.8
SERIAL_TIMEOUT_S = 1.0

CALIB_FILE = "calib.json"

# ==== КАМЕРА ====
USE_PICAMERA2 = True
try:
    from picamera2 import Picamera2
except Exception:
    USE_PICAMERA2 = False

def open_camera():
    if USE_PICAMERA2:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
        ))
        picam2.start()
        def read():
            frame = picam2.capture_array()
            return frame
        return read, None
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        def read():
            ok, frame = cap.read()
            if not ok: return None
            return frame
        return read, cap

def rotate_frame(frame, deg):
    if deg == 0: return frame
    if deg == 90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# ==== СЕРИАЛ ====
def find_serial_port():
    # сначала фиксированные кандидаты
    for p in SERIAL_PORT_CANDIDATES:
        if os.path.exists(p):
            return p
    # затем по описанию
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "Arduino" in p.description or "wchusb" in p.description.lower():
            return p.device
    return None

class ArduinoLink:
    def __init__(self, port, baud=SERIAL_BAUD):
        self.ser = serial.Serial(port, baudrate=baud, timeout=SERIAL_TIMEOUT_S)
        time.sleep(2.0)  # ардуино ресет
        self.lock = threading.Lock()
        self.last_beat = time.time()

    def cmd(self, txt):
        with self.lock:
            self.ser.write((txt.strip() + "\n").encode("ascii"))

    def heartbeat(self):
        now = time.time()
        if now - self.last_beat >= HEARTBEAT_PERIOD:
            self.cmd("PING")
            self.last_beat = now

    def set_angles(self, pan, tilt):
        pan = max(PAN_MIN, min(PAN_MAX, int(round(pan))))
        tilt = max(TILT_MIN, min(TILT_MAX, int(round(tilt))))
        self.cmd(f"SET {pan} {tilt}")

    def laser(self, on):
        self.cmd("LASER 1" if on else "LASER 0")

    def home(self):
        self.cmd("HOME")

# ==== КАЛИБРОВКА ====
def detect_red_laser(frame_bgr):
    # поищем ярко-красную точку (две зоны HSV)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 120, 150), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 150), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 5:  # отсекаем шум
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    return (cx, cy)

def fit_affine(src_pts, dst_pts):
    # строим аффинное преобразование: [x y 1] * W = [a b] (подгон по МНК)
    # здесь строим обе модели: (углы)->(пиксели) и (пиксели)->(углы)
    A = np.hstack([src_pts, np.ones((src_pts.shape[0],1))])  # N x 3
    B = dst_pts  # N x 2
    W, *_ = np.linalg.lstsq(A, B, rcond=None)  # 3x2
    return W  # примен: P = [pan tilt 1] @ W

def invert_mapping(angles_to_px_W, sample_center):
    # Можно прямо обучить обратную модель по собранным точкам,
    # но иногда достаточно аналитического псевдообратного.
    return np.linalg.pinv(angles_to_px_W)

def save_calib(angles_to_px_W, px_to_angles_W):
    data = {
        "angles_to_px_W": angles_to_px_W.tolist(),
        "px_to_angles_W": px_to_angles_W.tolist(),
        "frame_size": [FRAME_W, FRAME_H],
        "rotate_deg": ROTATE_DEG,
        "pan_limits": [PAN_MIN, PAN_MAX],
        "tilt_limits": [TILT_MIN, TILT_MAX]
    }
    with open(CALIB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_calib():
    with open(CALIB_FILE, "r") as f:
        d = json.load(f)
    return (np.array(d["angles_to_px_W"]),
            np.array(d["px_to_angles_W"]),
            tuple(d["frame_size"]))

def run_calibration():
    read, cap = open_camera()
    port = find_serial_port()
    assert port, "Не найден сериал-порт Arduino"
    ard = ArduinoLink(port)
    ard.home(); time.sleep(0.8)
    ard.laser(False)

    print("[CAL] Начало сканирования… Поставьте систему на ровную стену 1.5–3 м.")
    # сетка углов
    pan_vals = np.linspace(PAN_MIN+5, PAN_MAX-5, 8, dtype=int)
    tilt_vals = np.linspace(TILT_MIN+5, TILT_MAX-5, 6, dtype=int)

    samples_angles = []
    samples_pixels = []

    try:
        for t in tilt_vals:
            for p in pan_vals if ((np.where(tilt_vals==t)[0][0])%2==0) else pan_vals[::-1]:
                ard.set_angles(p, t)
                time.sleep(0.18)
                ard.laser(True)
                dot = None
                for _ in range(6):
                    frame = read()
                    if frame is None: continue
                    frame = rotate_frame(frame, ROTATE_DEG)
                    dot = detect_red_laser(frame)
                    if dot: break
                    time.sleep(0.05)
                ard.laser(False)
                if dot:
                    samples_angles.append([p, t])
                    samples_pixels.append([dot[0], dot[1]])
                    print(f"[CAL] {p:3d},{t:3d} -> px {dot}")
                else:
                    print(f"[CAL] {p:3d},{t:3d} -> нет точки (пропуск)")
                ard.heartbeat()
        if len(samples_angles) < 8:
            raise RuntimeError("Слишком мало калибровочных точек. Повторите.")
        A = np.array(samples_angles, dtype=float)
        P = np.array(samples_pixels, dtype=float)

        W_ang2px = fit_affine(A, P)                 # (3x2)
        # для обратной лучше обучить отдельно: [x y 1] -> [pan tilt]
        X = np.hstack([P, np.ones((P.shape[0],1))]) # N x 3
        Y = A                                       # N x 2
        W_px2ang, *_ = np.linalg.lstsq(X, Y, rcond=None)

        save_calib(W_ang2px, W_px2ang)
        print("[CAL] Готово. Калибровка записана в", CALIB_FILE)
    finally:
        if cap is not None: cap.release()

# ==== ТРЕКИНГ ====
class AimController:
    def __init__(self, ard, W_px2ang):
        self.ard = ard
        self.W_px2ang = W_px2ang
        self.target_px_ema = None
        self.last_seen = 0
        self.pan = PAN_HOME
        self.tilt = TILT_HOME
        self.scan_dir = 1
        self.scan_phase = 0.0
        self.holding = False

    def pix_to_angles(self, x, y):
        X = np.array([x, y, 1.0])
        pan, tilt = X @ self.W_px2ang  # [3]x[3x2] = [2]
        return float(pan), float(tilt)

    def update(self, has_target, target_px):
        now = time.time()
        self.ard.heartbeat()

        if has_target:
            self.last_seen = now
            if self.target_px_ema is None:
                self.target_px_ema = np.array(target_px, dtype=float)
            else:
                self.target_px_ema = (1-EMA_ALPHA)*self.target_px_ema + EMA_ALPHA*np.array(target_px, dtype=float)
            x, y = self.target_px_ema
            pan_cmd, tilt_cmd = self.pix_to_angles(x, y)
            # Ограничения
            pan_cmd = max(PAN_MIN, min(PAN_MAX, pan_cmd))
            tilt_cmd = max(TILT_MIN, min(TILT_MAX, tilt_cmd))
            self.pan, self.tilt = pan_cmd, tilt_cmd
            self.ard.set_angles(self.pan, self.tilt)

            # точность удержания
            cx, cy = FRAME_W/2, FRAME_H/2
            err = math.hypot(x - cx, y - cy)
            self.holding = err <= PIX_TOLERANCE
            self.ard.laser(self.holding)
        else:
            # Потеря цели
            if now - self.last_seen > LOST_TARGET_TIMEOUT:
                self.target_px_ema = None
                self.holding = False
                self.ard.laser(False)
                # режим сканирования
                self.scan_phase += SCAN_SPEED_DEG * self.scan_dir
                if self.scan_phase > (PAN_MAX - PAN_MIN) * 0.9 or self.scan_phase < 0:
                    self.scan_dir *= -1
                    self.scan_phase = max(0, min(self.scan_phase, (PAN_MAX - PAN_MIN) * 0.9))
                pan_cmd = PAN_MIN + self.scan_phase
                # небольшое покачивание по наклону
                t = time.time()
                tilt_cmd = PAN_HOME*0 + TILT_HOME + SCAN_TILT_SWING * math.sin(t*0.8)
                tilt_cmd = max(TILT_MIN, min(TILT_MAX, tilt_cmd))
                self.pan, self.tilt = pan_cmd, tilt_cmd
                self.ard.set_angles(self.pan, self.tilt)

def run_tracking(target_labels):
    # модель
    model = YOLO(MODEL_PATH)
    read, cap = open_camera()
    port = find_serial_port()
    assert port, "Не найден сериал-порт Arduino"
    ard = ArduinoLink(port)
    ard.home(); time.sleep(0.8)
    ard.laser(False)

    # калибровка
    W_ang2px, W_px2ang, (fw, fh) = load_calib()
    assert fw==FRAME_W and fh==FRAME_H, "Размер кадра изменился с момента калибровки – откалибруйте заново."

    aim = AimController(ard, W_px2ang)

    label_set = set([s.strip().lower() for s in target_labels])

    print("[TRACK] Старт. Поиск меток:", label_set)
    try:
        while True:
            frame = read()
            if frame is None: continue
            frame = rotate_frame(frame, ROTATE_DEG)

            # детекция
            results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF_THRES, verbose=False)
            has_target = False
            target_center = None

            r0 = results[0]
            for b in r0.boxes:
                cls_id = int(b.cls)
                name = r0.names[cls_id].lower()
                if target_labels and name not in label_set:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                has_target = True
                target_center = (cx, cy)
                break  # берём первую подходящую цель

            aim.update(has_target, target_center if target_center else (0,0))

            # отладочное окно (опционально)
            # рисуем маркер центра
            # cv2.circle(frame, (FRAME_W//2, FRAME_H//2), 6, (255,255,255), 2)
            # if target_center:
            #     cv2.circle(frame, target_center, 6, (0,255,0), 2)
            # cv2.imshow("track", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break
    finally:
        ard.laser(False)
        if cap is not None: cap.release()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="Запустить самокалибровку по красной точке лазера")
    ap.add_argument("--track", action="store_true", help="Запустить распознавание и наведение")
    ap.add_argument("--labels", type=str, default="person", help="Список меток через запятую (пример: person,cup,bottle)")
    args = ap.parse_args()

    if args.calibrate:
        run_calibration()
    elif args.track:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
        run_tracking(labels)
    else:
        print("Укажите --calibrate или --track")
