#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2, time, math, numpy as np
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH   = "yolov8n.pt"
CONF_THRES   = 0.25
ONLY         = ["cup"]            # какие классы ловим (имена COCO)
IMGSZ        = 320                # вход модели (меньше => быстрее)
FRAME_SIZE   = (640, 480)         # реальное изображение с камеры
ROTATE_DEG   = 180                # 0/90/180/270

# Параметры наведения
FOV_H_DEG    = 66.0               # CM3 ~66° по горизонтали
FOV_V_DEG    = 41.0               # CM3 ~41° по вертикали
Kp_deg       = 0.6                # чувствительность (град/град)
DEADBAND_PX  = 4                  # мёртвая зона по пикселям
SMOOTH       = 0.6                # 0..1, больше = плавнее (экспон. сглаживание)
MAX_STEP_DEG = 4.0                # ограничение скорости шаг/цикл

# Ограничения серв и старт
MIN_ANGLE    = 30
MAX_ANGLE    = 150
servo_x      = 90.0               # старт по оси X (yaw)
servo_y      = 90.0               # старт по оси Y (pitch)

# Последовательный порт
BAUDRATE     = 115200             # советую 115200
RECONNECT_S  = 2.0                # период попытки переподключения

# =============================


def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        # фильтры на всякий случай
        if "USB" in p.device or "ACM" in p.device:
            return p.device
    return None


def open_serial():
    port = find_arduino_port()
    if not port:
        return None
    try:
        s = serial.Serial(port, BAUDRATE, timeout=0.05)
        time.sleep(1.8)  # подождать перезапуск Arduino
        print(f"[SERIAL] connected: {port}")
        return s
    except Exception as e:
        print(f"[SERIAL] open error: {e}")
        return None


def send_angles(ser, x, y):
    """Отправка углов в градусах, формат X:..,Y:.. (CRLF)."""
    if ser is None:
        return
    try:
        ser.write(f"X:{int(x)},Y:{int(y)}\r\n".encode())
    except Exception as e:
        print(f"[SERIAL] write error: {e}")


def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))


def largest_of_classes(results, wanted_ids):
    """Вернёт центр (cx,cy) самой крупной нужной цели или None."""
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
    """Поиск красной точки лазера (просто и быстро, можно подстроить под ваш цвет)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # два диапазона красного
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
sx, sy = 0.0, 0.0  # сглаженные угловые ошибки

print("Нажмите 'q' для выхода")
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

        # инференс
        res = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        obj = largest_of_classes(res, wanted_ids)
        laser = detect_laser(frame)

        # Источник ошибки:
        # 1) если виден лазер и объект — ведём лазер к объекту (закрытая петля),
        # 2) иначе — ведём центр турели к объекту (от центра кадра).
        target_for_error = None
        ref_point = None
        if obj is not None and laser is not None:
            target_for_error = obj
            ref_point = laser
            last_seen = time.time()
            auto_centered = False
        elif obj is not None:
            target_for_error = obj
            ref_point = frame_center
            last_seen = time.time()
            auto_centered = False

        if target_for_error is not None:
            dx_px = target_for_error[0] - ref_point[0]
            dy_px = target_for_error[1] - ref_point[1]

            # мёртвая зона по пикселям
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            # пиксели → градусы
            err_yaw_deg   =  (dx_px / w) * FOV_H_DEG
            err_pitch_deg = -(dy_px / h) * FOV_V_DEG

            # сгладим
            sx = SMOOTH * sx + (1.0 - SMOOTH) * err_yaw_deg
            sy = SMOOTH * sy + (1.0 - SMOOTH) * err_pitch_deg

            # ограничим скорость (на случай больших рывков)
            step_x = clamp(sx * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sy * Kp_deg, -MAX_STEP_DEG, MAX_STEP_DEG)

            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)

            send_angles(ser, servo_x, servo_y)

        # если цель потеряна — мягкий возврат в центр
        elif (time.time() - last_seen) > 1.5 and not auto_centered:
            servo_x = 90.0; servo_y = 90.0
            send_angles(ser, servo_x, servo_y)
            auto_centered = True

        # авто-переподключение последовательного порта
        if ser is None or not ser.is_open:
            if (time.time() - last_seen) > RECONNECT_S:
                ser = open_serial()

        # Рисуем оверлей
        if obj:   cv2.circle(frame, obj,   5, (255, 0, 0), -1)   # цель (синяя)
        if laser: cv2.circle(frame, laser, 5, (0, 0, 255), -1)   # лазер (красная)
        cv2.circle(frame, frame_center, 4, (0, 255, 0), 1)

        frames += 1
        if frames % 10 == 0:
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0
            cv2.putText(frame, f"FPS:{fps:.1f}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Auto Aim Tracker", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    try:
        picam2.stop()
    except: pass
    cv2.destroyAllWindows()
    if ser:
        try:
            ser.close()
        except: pass
