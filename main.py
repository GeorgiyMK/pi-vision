#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2, time, numpy as np
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH   = "yolov8n.pt"
CONF_THRES   = 0.25
ONLY         = ["cup"]            # какие классы ловим (имена COCO)
IMGSZ        = 320
FRAME_SIZE   = (640, 480)
ROTATE_DEG   = 180

# управление осями
INVERT_X     = False
INVERT_Y     = False

# поле зрения камеры
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
# =============================


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
    if ser:
        ser.write(f"X:{int(x)},Y:{int(y)}\r\n".encode())


def detect_laser(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 200), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 3: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))


def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))


# === КАМЕРА + МОДЕЛЬ ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": FRAME_SIZE}))
picam2.start()

model = YOLO(MODEL_PATH)
name2id = {v: k for k, v in model.names.items()}
wanted_ids = [name2id[n] for n in ONLY if n in name2id]

ser = open_serial()
frame_center = (FRAME_SIZE[0]//2, FRAME_SIZE[1]//2)

# === КАЛИБРОВКА ЛАЗЕРА ===
print("Калибровка лазера... убедись, что лазер включен и светит на экран камеры")
time.sleep(2.0)
calib_frame = picam2.capture_array()
laser_pos = detect_laser(calib_frame)
if laser_pos is None:
    print("⚠ Лазер не найден, используем центр")
    laser_offset = (0, 0)
else:
    laser_offset = (laser_pos[0] - frame_center[0],
                    laser_pos[1] - frame_center[1])
    print(f"Лазер найден, смещение = {laser_offset}")

# Переменные
last_seen = 0.0
auto_centered = False
sx, sy = 0.0, 0.0

print("Нажмите 'q' для выхода")
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

        # инференс
        res = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        obj = None
        if res and len(res[0].boxes) > 0:
            boxes = res[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in wanted_ids: continue
                if float(boxes.conf[i].item()) < CONF_THRES: continue
                x1,y1,x2,y2 = boxes.xyxy[i].tolist()
                obj = (int((x1+x2)/2), int((y1+y2)/2))
                break

        # включаем/выключаем лазер
        if obj is not None:
            if ser: ser.write(b"LAS 1\r\n")
        else:
            if ser: ser.write(b"LAS 0\r\n")

        if obj is not None:
            dx_px = (obj[0] - frame_center[0]) - laser_offset[0]
            dy_px = (obj[1] - frame_center[1]) - laser_offset[1]

            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            err_yaw_deg   =  (dx_px / w) * FOV_H_DEG
            err_pitch_deg = -(dy_px / h) * FOV_V_DEG

            sx = SMOOTH * sx + (1.0 - SMOOTH) * err_yaw_deg
            sy = SMOOTH * sy + (1.0 - SMOOTH) * err_pitch_deg

            step_x = clamp(sx * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sy * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)

            if INVERT_X: step_x = -step_x
            if INVERT_Y: step_y = -step_y

            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)

            send_angles(ser, servo_x, servo_y)
            last_seen = time.time()
            auto_centered = False

        elif (time.time() - last_seen) > 1.5 and not auto_centered:
            servo_x = 90.0; servo_y = 90.0
            send_angles(ser, servo_x, servo_y)
            auto_centered = True

        # рисуем
        if obj:   cv2.circle(frame, obj, 5, (255, 0, 0), -1)
        if laser_pos: cv2.circle(frame, (frame_center[0]+laser_offset[0],
                                         frame_center[1]+laser_offset[1]),
                                 5, (0, 0, 255), -1)
        cv2.circle(frame, frame_center, 4, (0, 255, 0), 1)

        cv2.imshow("Auto Aim Tracker", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    if ser: ser.close()
