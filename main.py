#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2, time, numpy as np
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH   = "yolov8n.pt"
CONF_THRES   = 0.25
ONLY         = ["cup"]
IMGSZ        = 320
FRAME_SIZE   = (640, 480)
ROTATE_DEG   = 180

FOV_H_DEG    = 66.0
FOV_V_DEG    = 41.0
Kp_deg       = 0.6
DEADBAND_PX  = 4
SMOOTH       = 0.6
MAX_STEP_DEG = 4.0

MIN_ANGLE    = 30
MAX_ANGLE    = 150
servo_x      = 90.0
servo_y      = 90.0

BAUDRATE     = 115200
RECONNECT_S  = 2.0

# ========== ФУНКЦИИ ==========
def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        if "USB" in p.device or "ACM" in p.device:
            return p.device
    return None

def open_serial():
    port = find_arduino_port()
    if not port:
        return None
    try:
        s = serial.Serial(port, BAUDRATE, timeout=0.05)
        time.sleep(1.8)
        print(f"[SERIAL] connected: {port}")
        return s
    except Exception as e:
        print(f"[SERIAL] open error: {e}")
        return None

def send_angles(ser, x, y):
    if ser: ser.write(f"X:{int(x)},Y:{int(y)}\r\n".encode())

def send_laser(ser, state):
    """Вкл/выкл лазера (1/0)."""
    if ser:
        ser.write(f"LAS {1 if state else 0}\r\n".encode())

def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))

def largest_of_classes(results, wanted_ids):
    if not results or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    pick = None
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if wanted_ids and cls_id not in wanted_ids:
            continue
        if float(boxes.conf[i].item()) < CONF_THRES:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        area = (x2 - x1) * (y2 - y1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        if (pick is None) or (area > pick[-1]):
            pick = (int(cx), int(cy), area)
    return None if pick is None else (pick[0], pick[1])

def detect_laser(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 200), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

# === КАМЕРА + МОДЕЛЬ ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}))
picam2.start()

model = YOLO(MODEL_PATH)
name2id = {v: k for k, v in model.names.items()}
wanted_ids = [name2id[n] for n in ONLY if n in name2id]

ser = open_serial()
frame_center = (FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2)
last_seen = 0.0
auto_centered = False
sx, sy = 0.0, 0.0

laser_state = False  # текущее состояние лазера

print("Нажмите 'q' для выхода, 'L' для вкл/выкл лазера")
t0 = time.time(); frames = 0
try:
    while True:
        frame = picam2.capture_array()

        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]
        res = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        obj = largest_of_classes(res, wanted_ids)
        laser = detect_laser(frame)

        # Ведение цели
        target_for_error = None
        ref_point = None
        if obj and laser:
            target_for_error = obj
            ref_point = laser
            last_seen = time.time()
            auto_centered = False
        elif obj:
            target_for_error = obj
            ref_point = frame_center
            last_seen = time.time()
            auto_centered = False

        if target_for_error:
            dx_px = target_for_error[0] - ref_point[0]
            dy_px = target_for_error[1] - ref_point[1]
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0
            err_yaw_deg   =  (dx_px / w) * FOV_H_DEG
            err_pitch_deg = -(dy_px / h) * FOV_V_DEG
            sx = SMOOTH * sx + (1.0 - SMOOTH) * err_yaw_deg
            sy = SMOOTH * sy + (1.0 - SMOOTH) * err_pitch_deg
            step_x = clamp(sx * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sy * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

        elif (time.time() - last_seen) > 1.5 and not auto_centered:
            servo_x = 90.0; servo_y = 90.0
            send_angles(ser, servo_x, servo_y)
            auto_centered = True

        # Авто-переподключение
        if ser is None or not ser.is_open:
            if (time.time() - last_seen) > RECONNECT_S:
                ser = open_serial()

        # Рисуем
        if obj:   cv2.circle(frame, obj,   5, (255, 0, 0), -1)
        if laser: cv2.circle(frame, laser, 5, (0, 0, 255), -1)
        cv2.circle(frame, frame_center, 4, (0, 255, 0), 1)

        frames += 1
        if frames % 10 == 0:
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0
            cv2.putText(frame, f"FPS:{fps:.1f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Auto Aim Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            laser_state = not laser_state
            send_laser(ser, laser_state)
            print("Laser ON" if laser_state else "Laser OFF")

except KeyboardInterrupt:
    pass
finally:
    try: picam2.stop()
    except: pass
    cv2.destroyAllWindows()
    if ser:
        try: ser.close()
        except: pass
