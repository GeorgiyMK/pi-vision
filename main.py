#!/usr/bin/env python3
from picamera2 import Picamera2
import cv2, time, json, os, numpy as np
import serial, serial.tools.list_ports
from ultralytics import YOLO

# ====== НАСТРОЙКИ ======
MODEL_PATH   = "yolov8n.pt"
ONLY         = ["cup"]          # классы YOLO
CONF_THRES   = 0.35
IMGSZ        = 320
FRAME_SIZE   = (640, 480)
ROTATE_DEG   = 180

FOV_H_DEG    = 66.0             # CM3 ориентир
FOV_V_DEG    = 41.0
Kp_deg       = 0.7              # скорость реакции
SMOOTH       = 0.6
DEADBAND_PX  = 4
MAX_STEP_DEG = 5.0

MIN_ANGLE    = 30
MAX_ANGLE    = 150
BAUDRATE     = 115200
CALIB_PATH   = "calib.json"

# «обучение» оффсетов, если есть постоянная ошибка
Ki_bias      = 0.03
BIAS_LIMIT   = 15.0

# тайминги/состояния
MISS_DOT_TO_SEARCH = 12         # кадров без точки до режима SEARCH
SEARCH_RADIUS_STEP = 2.5        # шаг радиуса спирали (градусы за цикл)
SEARCH_ANG_STEP    = 18         # шаг угла спирали (градусы за цикл)
RECONNECT_S        = 2.0

# ====== УТИЛИТЫ ======
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

def send_laser(ser, on):
    if ser:
        ser.write(("LAS 1\r\n" if on else "LAS 0\r\n").encode())

def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))

def detect_laser(frame):
    # более терпимый поиск красной точки
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0,  80, 180), (12, 255, 255))
    m2 = cv2.inRange(hsv, (168, 80, 180), (180,255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 2: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

def largest_target(results, wanted_ids, conf=0.35):
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
        if (pick is None) or (area>pick[-1]): pick = (cx, cy, area)
    return None if pick is None else (pick[0], pick[1])

def load_calib():
    if os.path.exists(CALIB_PATH):
        try:
            with open(CALIB_PATH,"r") as f: return json.load(f)
        except: pass
    return {"invert_x": False, "invert_y": False, "bias_x": 0.0, "bias_y": 0.0}

def save_calib(c):
    with open(CALIB_PATH,"w") as f: json.dump(c, f, indent=2)
    print(f"[CALIB] saved: {c}")

# ====== ИНИЦ ======
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888","size":FRAME_SIZE}))
picam2.start()

# (опционально поменьше экспозицию, чтобы точка не пересвечивалась)
# picam2.set_controls({"AeEnable": True, "ExposureTime": 5000})

model = YOLO(MODEL_PATH)
name2id = {v:k for k,v in model.names.items()}
wanted_ids = [name2id[n] for n in ONLY if n in name2id]

ser = open_serial()

w, h = FRAME_SIZE
cx0, cy0 = w//2, h//2
servo_x, servo_y = 90.0, 90.0
laser_on = False

cal = load_calib()
invert_x = cal["invert_x"]
invert_y = cal["invert_y"]
bias_x_deg = float(cal["bias_x"])
bias_y_deg = float(cal["bias_y"])

sx_f, sy_f = 0.0, 0.0
miss_dot = 0

state = "IDLE"  # IDLE / CHASE / SEARCH / TRACK
search_radius = 0.0
search_angle  = 0.0

print("Клавиши: L — лазер, C — центр, I/K — инверт X/Y, S — сохранить, Q — выход.")
try:
    while True:
        frame = picam2.capture_array()
        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        res  = model.predict(frame, conf=CONF_THRES, imgsz=IMGSZ, verbose=False)
        obj  = largest_target(res, wanted_ids, CONF_THRES)
        dot  = detect_laser(frame)

        # выбор состояния
        prev_state = state
        if obj and dot:
            state = "TRACK"
            miss_dot = 0
        elif obj and not dot:
            miss_dot += 1
            if miss_dot >= MISS_DOT_TO_SEARCH:
                state = "SEARCH"
            else:
                state = "CHASE"
        else:
            state = "IDLE"
            miss_dot = 0
            search_radius = 0.0
            search_angle  = 0.0

        if state != prev_state:
            print(f"[STATE] {prev_state} -> {state}")

        # поведение в состояниях
        if state == "TRACK":
            # ошибка между точкой и объектом
            dx_px = obj[0] - dot[0]
            dy_px = obj[1] - dot[1]
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            err_x =  (dx_px / w) * FOV_H_DEG
            err_y = -(dy_px / h) * FOV_V_DEG

            sx_f = SMOOTH*sx_f + (1-SMOOTH)*err_x
            sy_f = SMOOTH*sy_f + (1-SMOOTH)*err_y

            sgn_x = -1 if invert_x else 1
            sgn_y = -1 if invert_y else 1

            step_x = clamp(sgn_x*Kp_deg*sx_f + bias_x_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sgn_y*Kp_deg*sy_f + bias_y_deg, -MAX_STEP_DEG, MAX_STEP_DEG)

            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

            # подучиваем оффсет, чтобы средняя ошибка стремилась к нулю
            bias_x_deg = clamp(bias_x_deg + Ki_bias*sx_f, -BIAS_LIMIT, BIAS_LIMIT)
            bias_y_deg = clamp(bias_y_deg + Ki_bias*sy_f, -BIAS_LIMIT, BIAS_LIMIT)

            if not laser_on:
                send_laser(ser, True); laser_on = True

        elif state == "CHASE":
            # ведём «на глаз» — ошибку считаем от центра (где «должна» быть точка)
            dx_px = obj[0] - cx0
            dy_px = obj[1] - cy0
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            err_x =  (dx_px / w) * FOV_H_DEG
            err_y = -(dy_px / h) * FOV_V_DEG

            sx_f = SMOOTH*sx_f + (1-SMOOTH)*err_x
            sy_f = SMOOTH*sy_f + (1-SMOOTH)*err_y

            sgn_x = -1 if invert_x else 1
            sgn_y = -1 if invert_y else 1

            step_x = clamp(sgn_x*Kp_deg*sx_f + bias_x_deg, -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(sgn_y*Kp_deg*sy_f + bias_y_deg, -MAX_STEP_DEG, MAX_STEP_DEG)

            servo_x = clamp(servo_x - step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

            if not laser_on:
                send_laser(ser, True); laser_on = True

        elif state == "SEARCH":
            # спиральный поиск вокруг предполагаемой точки (центр кадра)
            if not laser_on:
                send_laser(ser, True); laser_on = True

            search_radius += SEARCH_RADIUS_STEP
            search_angle  = (search_angle + SEARCH_ANG_STEP) % 360
            # смещения в градусах
            dx_deg = (search_radius * np.cos(np.deg2rad(search_angle)))
            dy_deg = (search_radius * np.sin(np.deg2rad(search_angle)))

            # применим как шаги серв (учитывая инверсию)
            sgn_x = -1 if invert_x else 1
            sgn_y = -1 if invert_y else 1
            servo_x = clamp(servo_x - sgn_x*dx_deg, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + sgn_y*dy_deg, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

            # ограничим радиус, чтобы не упереться в механику
            if search_radius > 20: search_radius = 0

        else:  # IDLE
            if laser_on: send_laser(ser, False); laser_on = False
            # плавно к центру
            servo_x = clamp(servo_x + (90-servo_x)*0.1, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + (90-servo_y)*0.1, MIN_ANGLE, MAX_ANGLE)
            send_angles(ser, servo_x, servo_y)

        # отрисовка
        if obj: cv2.circle(frame, obj, 5, (255,0,0), -1)
        if dot: cv2.circle(frame, dot, 5, (0,0,255), -1)
        cv2.circle(frame, (cx0,cy0), 4, (0,255,0), 1)
        cv2.putText(frame, f"{state}  INVX:{int(invert_x)} INVY:{int(invert_y)}  Bx:{bias_x_deg:.1f} By:{bias_y_deg:.1f}",
                    (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
        cv2.imshow("Auto Aim (seek+track)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('l'):
            laser_on = not laser_on; send_laser(ser, laser_on)
        elif key == ord('c'):
            servo_x, servo_y = 90.0, 90.0; send_angles(ser, servo_x, servo_y)
        elif key == ord('i'):
            invert_x = not invert_x; print("invert_x =", invert_x)
        elif key == ord('k'):
            invert_y = not invert_y; print("invert_y =", invert_y)
        elif key == ord('s'):
            save_calib({"invert_x":invert_x, "invert_y":invert_y,
                        "bias_x":bias_x_deg, "bias_y":bias_y_deg})

except KeyboardInterrupt:
    pass
finally:
    try:
        if ser: send_laser(ser, False)
    except: pass
    if ser:
        try: ser.close()
        except: pass
    try: picam2.stop()
    except: pass
    cv2.destroyAllWindows()
    save_calib({"invert_x":invert_x, "invert_y":invert_y,
                "bias_x":bias_x_deg, "bias_y":bias_y_deg})
