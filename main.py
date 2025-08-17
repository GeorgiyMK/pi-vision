#!/usr/bin/env python3
# Auto-aim with adaptive calibration: laser dot + object
from picamera2 import Picamera2
import cv2, time, json, os, numpy as np
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ========= НАСТРОЙКИ =========
MODEL_PATH   = "yolov8n.pt"
ONLY         = ["cup"]          # какие классы ловим (имена COCO)
CONF_THRES   = 0.30
IMGSZ        = 320
FRAME_SIZE   = (640, 480)
ROTATE_DEG   = 180              # 0/90/180/270

# Поле зрения камеры (CM3 ориентир)
FOV_H_DEG    = 66.0
FOV_V_DEG    = 41.0

# Управление
Kp_deg       = 0.6              # пропорциональная «чувствительность» в градусах
Ki_bias      = 0.04             # «учёба» оффсета (на каждом кадре), 0..0.1
SMOOTH       = 0.6              # сглаживание ошибок (0 — резко, 0.9 — плавно)
DEADBAND_PX  = 4                # мёртвая зона по пикселям
MAX_STEP_DEG = 5.0              # максимум изменения угла за кадр

# Пределы серв (безопасные)
MIN_ANGLE    = 30
MAX_ANGLE    = 150

# Последовательный порт
BAUDRATE     = 115200
RECONNECT_S  = 2.0

# Файл калибровки
CALIB_PATH   = "calib.json"
# =============================

def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        if "USB" in p.device or "ACM" in p.device:
            return p.device
    return None

def open_serial():
    port = find_arduino_port()
    if not port: return None
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

def send_laser(ser, on: bool):
    if ser:
        ser.write(("LAS 1\r\n" if on else "LAS 0\r\n").encode())

def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))

def detect_laser(frame):
    """Поиск красной точки лазера, быстрый HSV-фильтр."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 200), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 200), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    # Чуть шумоподавления
    mask = cv2.medianBlur(mask, 3)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 3: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def largest_target(results, wanted_ids, conf=0.3):
    if not results or len(results[0].boxes)==0: return None
    boxes = results[0].boxes
    pick = None
    for i in range(len(boxes)):
        cid = int(boxes.cls[i].item())
        if wanted_ids and cid not in wanted_ids: continue
        if float(boxes.conf[i].item()) < conf: continue
        x1,y1,x2,y2 = boxes.xyxy[i].tolist()
        area = (x2-x1)*(y2-y1)
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        if (pick is None) or (area>pick[-1]):
            pick = (cx, cy, area)
    return None if pick is None else (pick[0], pick[1])

def load_calib():
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH,"r") as f: return json.load(f)
        except: pass
    return {"invert_x": False, "invert_y": False, "bias_x": 0.0, "bias_y": 0.0}

def save_calib(c):
    with open(CALIB_PATH,"w") as f: json.dump(c, f, indent=2)
    print(f"[CALIB] saved to {CALIB_PATH}: {c}")

# === ИНИЦИАЛИЗАЦИЯ КАМЕРЫ/МОДЕЛИ ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888","size":FRAME_SIZE}))
picam2.start()

model = YOLO(MODEL_PATH)
name2id = {v:k for k,v in model.names.items()}
wanted_ids = [name2id[n] for n in ONLY if n in name2id]

ser = open_serial()

# Состояние
w, h = FRAME_SIZE
cx0, cy0 = w//2, h//2
servo_x, servo_y = 90.0, 90.0
laser_on = False

cal = load_calib()
invert_x = cal["invert_x"]
invert_y = cal["invert_y"]
bias_x_deg = float(cal["bias_x"])   # смещение по X (в градусах)
bias_y_deg = float(cal["bias_y"])

# Для адаптивной инверсии
prev_err_x = None
prev_err_y = None
worse_x_count = 0
worse_y_count = 0
WORSE_N = 6          # сколько подряд «хуже», чтобы flip ось
MARGIN = 0.5         # допуск по градусам для сравнения «хуже/лучше»

# Ошибка с фильтром
sx_filt = 0.0
sy_filt = 0.0

print("Запуск. Горячие клавиши: L — лазер, C — центр, S — сохранить калибровку, R — сброс, Q — выход.")
try:
    while True:
        frame = picam2.capture_array()
        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Детекция
        res = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        obj = largest_target(res, wanted_ids, CONF_THRES)
        dot = detect_laser(frame)

        # Логика включения лазера
        want_laser = obj is not None
        if want_laser != laser_on:
            send_laser(ser, want_laser)
            laser_on = want_laser

        # Наведение (закрытая петля, если видим и лазер, и цель; иначе от центра)
        if obj:
            ref = dot if dot else (cx0, cy0)
            dx_px = obj[0] - ref[0]
            dy_px = obj[1] - ref[1]

            # мёртвая зона
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            # пиксели -> градусы
            err_x_deg =  (dx_px / w) * FOV_H_DEG
            err_y_deg = -(dy_px / h) * FOV_V_DEG

            # фильтр
            sx_filt = SMOOTH*sx_filt + (1-SMOOTH)*err_x_deg
            sy_filt = SMOOTH*sy_filt + (1-SMOOTH)*err_y_deg

            # адаптивная инверсия: если после шага ошибка стала только больше несколько кадров подряд — флип
            if prev_err_x is not None:
                if abs(sx_filt) > abs(prev_err_x) + MARGIN:
                    worse_x_count += 1
                else:
                    worse_x_count = max(0, worse_x_count-1)
                if worse_x_count >= WORSE_N:
                    invert_x = not invert_x
                    worse_x_count = 0
                    print(f"[CALIB] invert_x -> {invert_x}")

            if prev_err_y is not None:
                if abs(sy_filt) > abs(prev_err_y) + MARGIN:
                    worse_y_count += 1
                else:
                    worse_y_count = max(0, worse_y_count-1)
                if worse_y_count >= WORSE_N:
                    invert_y = not invert_y
                    worse_y_count = 0
                    print(f"[CALIB] invert_y -> {invert_y}")

            prev_err_x = sx_filt
            prev_err_y = sy_filt

            # шаги с учётом инверсии и оффсета
            sgn_x = -1 if invert_x else 1
            sgn_y = -1 if invert_y else 1

            step_x = clamp(sgn_x * Kp_deg * sx_filt + bias_x_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sgn_y * Kp_deg * sy_filt + bias_y_deg, -MAX_STEP_DEG, MAX_STEP_DEG)

            # интегрируем команду
            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)

            send_angles(ser, servo_x, servo_y)

            # адаптация оффсетов (медленно тянем к нулю ошибки)
            bias_x_deg = clamp(bias_x_deg + Ki_bias * sx_filt, -15, 15)
            bias_y_deg = clamp(bias_y_deg + Ki_bias * sy_filt, -15, 15)

        else:
            # нет цели — держим центр и гасим лазер (выше уже отправили)
            servo_x = clamp(servo_x + (90-servo_x)*0.1, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + (90-servo_y)*0.1, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

        # Оверлеи
        if obj: cv2.circle(frame, obj, 5, (255,0,0), -1)
        if dot: cv2.circle(frame, dot, 5, (0,0,255), -1)
        cv2.circle(frame, (cx0,cy0), 4, (0,255,0), 1)
        txt = f"INVX:{int(invert_x)} INVY:{int(invert_y)} Bx:{bias_x_deg:.1f} By:{bias_y_deg:.1f}"
        cv2.putText(frame, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imshow("Auto Aim (adaptive)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            laser_on = not laser_on
            send_laser(ser, laser_on)
        elif key == ord('c'):
            servo_x, servo_y = 90.0, 90.0
            send_angles(ser, servo_x, servo_y)
        elif key == ord('s'):
            save_calib({"invert_x":invert_x,"invert_y":invert_y,
                        "bias_x":bias_x_deg,"bias_y":bias_y_deg})
        elif key == ord('r'):
            invert_x = invert_y = False
            bias_x_deg = bias_y_deg = 0.0
            print("[CALIB] reset")

except KeyboardInterrupt:
    pass
finally:
    if ser:
        try: send_laser(ser, False)
        except: pass
        try: ser.close()
        except: pass
    try: picam2.stop()
    except: pass
    cv2.destroyAllWindows()
    # авто-сохранение на выход
    save_calib({"invert_x":invert_x,"invert_y":invert_y,
                "bias_x":bias_x_deg,"bias_y":bias_y_deg})
