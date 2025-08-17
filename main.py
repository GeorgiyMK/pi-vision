#!/usr/bin/env python3
# Raspberry Pi 5 + Camera Module 3 -> Arduino (2 сервопривода + лазер)
# Режим: ведём ЦЕНТР объекта в центр кадра (без обратной связи по пятну лазера)

import time, json, os
import cv2
import numpy as np
import serial, serial.tools.list_ports
from picamera2 import Picamera2
from ultralytics import YOLO

# ================== НАСТРОЙКИ ==================
MODEL_PATH      = "yolov8n.pt"
WANTED_CLASSES  = ["cup"]      # какие классы вести (можно пустой список = любые)
CONF_THRESHOLD  = 0.35
IMGSZ           = 320

FRAME_SIZE      = (640, 480)
ROTATE_DEG      = 180          # 0/90/180/270

# Поле зрения камеры (примерно для Camera Module 3)
FOV_H_DEG       = 66.0
FOV_V_DEG       = 41.0

# Диапазон сервоприводов (на Arduino интерпретируются как абсолютные 0..180)
MIN_ANGLE       = 30
MAX_ANGLE       = 150

# Управление
KP              = 0.20         # усиление (увеличить — быстрее, но легче перерегулирование)
DEADBAND_PX     = 10           # мёртвая зона по пикселям
MAX_STEP_DEG    = 2.5          # макс изменение угла за такт (град)
CMD_HZ          = 25.0         # частота отправки команд в Serial (Гц)

# Протокол к Arduino: "XY" (X:...,Y:...) или "ANG" (ANG yaw pitch)
PROTOCOL_MODE   = "XY"
DEBUG_TX        = False
# ===============================================


def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))


def find_port():
    for p in serial.tools.list_ports.comports():
        if "/ttyUSB" in p.device or "/ttyACM" in p.device:
            return p.device
    return None


def open_serial():
    port = find_port()
    if not port:
        print("[SERIAL] Arduino не найдена.")
        return None
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(1.8)  # дождаться автоперезапуска Arduino
        print(f"[SERIAL] Подключено: {port}")
        return ser
    except Exception as e:
        print(f"[SERIAL] Ошибка открытия {port}: {e}")
        return None


def raw_send(ser, line):
    if not ser: return
    if not line.endswith("\r\n"): line += "\r\n"
    if DEBUG_TX: print("[TX]", line.strip())
    try:
        ser.write(line.encode())
    except Exception as e:
        print("[SERIAL] Ошибка записи:", e)


def send_angles(ser, ax, ay):
    mode = PROTOCOL_MODE.upper()
    if mode == "XY":
        raw_send(ser, f"X:{int(ax)},Y:{int(ay)}")
    elif mode == "ANG":
        yaw = ax - 90.0
        pitch = ay - 90.0
        raw_send(ser, f"ANG {yaw:.1f} {pitch:.1f}")


def send_laser(ser, on):
    raw_send(ser, "LAS 1" if on else "LAS 0")


def select_target_stable(results, wanted_ids, prev_target):
    """Стабильный выбор цели: ближайшая к прошлой, иначе крупнейшая.
       Возвращает (cx,cy,x1,y1,x2,y2) или None."""
    if not results or len(results[0].boxes) == 0:
        return None
    cand = []
    for box in results[0].boxes:
        cls = int(box.cls.item())
        if wanted_ids and cls not in wanted_ids:
            continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cx,cy = (x1+x2)/2, (y1+y2)/2
        area = (x2-x1)*(y2-y1)
        cand.append((cx,cy,x1,y1,x2,y2,area))
    if not cand:
        return None
    if prev_target is None:
        best = max(cand, key=lambda t: t[6])  # крупнейшая
    else:
        px,py = prev_target[0], prev_target[1]
        best = min(cand, key=lambda t: (t[0]-px)**2 + (t[1]-py)**2 + 0.000001*(1e8/max(t[6],1)))
    return (int(best[0]), int(best[1]), int(best[2]), int(best[3]), int(best[4]), int(best[5]))


def main():
    # --- камера ---
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}
    ))
    picam2.start()

    # --- модель ---
    model = YOLO(MODEL_PATH)
    name2id = {v:k for k,v in model.names.items()}
    wanted_ids = [name2id[n] for n in WANTED_CLASSES if n in name2id]

    # --- serial ---
    ser = open_serial()

    # Текущее положение сервоприводов
    servo_x = 90.0
    servo_y = 90.0
    next_tx = 0.0
    laser_on = False

    prev_target = None

    print("Запущено. Клавиши: Q — выход, I/K — инверсия X/Y, U/J — +/-Kp.")
    invert_x = False
    invert_y = False

    while True:
        frame = picam2.capture_array()
        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]

        # Детекция
        res = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
        tgt = select_target_stable(res, wanted_ids, prev_target)
        prev_target = tgt

        if tgt:
            # Включаем лазер
            if not laser_on:
                send_laser(ser, True)
                laser_on = True

            cx,cy,x1,y1,x2,y2 = tgt

            # Ошибка по пикселям от центра кадра
            dx_px = cx - w/2
            dy_px = cy - h/2

            # Мёртвая зона
            if abs(dx_px) < DEADBAND_PX: dx_px = 0
            if abs(dy_px) < DEADBAND_PX: dy_px = 0

            # Перевод в угловую ошибку (для масштаба/знаков; можно отключить и работать в пикселях)
            err_yaw   =  (dx_px / (w/2)) * (FOV_H_DEG/2)   # вправо +
            err_pitch = -(dy_px / (h/2)) * (FOV_V_DEG/2)   # вверх +

            if invert_x: err_yaw = -err_yaw
            if invert_y: err_pitch = -err_pitch

            # Пропорциональный шаг и ограничение шага
            step_x = clamp(KP * err_yaw,   -MAX_STEP_DEG, MAX_STEP_DEG)
            step_y = clamp(KP * err_pitch, -MAX_STEP_DEG, MAX_STEP_DEG)

            # Новые углы (абсолютные)
            servo_x = clamp(servo_x + step_x, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + step_y, MIN_ANGLE, MAX_ANGLE)

            # Отправляем не чаще CMD_HZ
            now = time.time()
            if now >= next_tx:
                send_angles(ser, servo_x, servo_y)
                next_tx = now + 1.0 / CMD_HZ

            # Рисуем
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.drawMarker(frame, (int(w/2),int(h/2)), (255,255,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(frame, f"Kp={KP:.2f} invX={int(invert_x)} invY={int(invert_y)} {PROTOCOL_MODE}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if DEBUG_TX:
                print(f"[DBG] err=({err_yaw:+.2f},{err_pitch:+.2f}) step=({step_x:+.2f},{step_y:+.2f}) servo=({servo_x:.1f},{servo_y:.1f})")
        else:
            if laser_on:
                send_laser(ser, False)
                laser_on = False
            # Мягко возвращаемся к центру (опционально)
            servo_x = clamp(servo_x + (90 - servo_x)*0.05, MIN_ANGLE, MAX_ANGLE)
            servo_y = clamp(servo_y + (90 - servo_y)*0.05, MIN_ANGLE, MAX_ANGLE)
            now = time.time()
            if now >= next_tx:
                send_angles(ser, servo_x, servo_y)
                next_tx = now + 1.0 / CMD_HZ
            cv2.drawMarker(frame, (int(w/2),int(h/2)), (255,255,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(frame, "no target", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow("Auto Aim (centering)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            invert_x = not invert_x; print("invert_x:", invert_x)
        elif key == ord('k'):
            invert_y = not invert_y; print("invert_y:", invert_y)
        elif key == ord('u'):
            KP += 0.03; print(f"Kp={KP:.2f}")
        elif key == ord('j'):
            KP = max(0.0, KP-0.03); print(f"Kp={KP:.2f}")

    # cleanup
    try: send_laser(ser, False)
    except: pass
    cv2.destroyAllWindows()
    picam2.stop()
    if ser: ser.close()


if __name__ == "__main__":
    main()
