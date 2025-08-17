#!/usr/bin/env python3
import cv2
import time
import serial
import serial.tools.list_ports
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO


# --- настройки ---
MODEL_PATH = "yolov8n.pt"
WANTED_CLASSES = ["cup"]   # отслеживаемый объект
CONF_THRESHOLD = 0.35
IMGSZ = 320
FRAME_SIZE = (640, 480)
ROTATE_DEG = 180

# --- сервы ---
servo_x = 90
servo_y = 90
ALPHA = 0.3         # сглаживание (0..1)
STEP = 2            # макс. шаг (градусов) за кадр
DEAD_MARGIN = 0.15  # зона допуска (нормированные координаты)
MIN_ANGLE = 20
MAX_ANGLE = 160

# --- связь ---
BAUDRATE = 115200


def connect_serial():
    """Автопоиск и подключение к Arduino"""
    try:
        port = next((p.device for p in serial.tools.list_ports.comports()
                    if "USB" in p.device or "ACM" in p.device), None)
        if not port:
            print("[SERIAL] Arduino не найдена.")
            return None
        ser = serial.Serial(port, BAUDRATE, timeout=0.1)
        time.sleep(2)
        print(f"[SERIAL] Подключено к {port}")
        return ser
    except Exception as e:
        print(f"[SERIAL] Ошибка: {e}")
        return None


def send_angles(ser, x, y):
    """Отправка углов на Arduino"""
    try:
        ser.write(f"X:{int(x)},Y:{int(y)}\r\n".encode())
    except:
        pass


def send_laser(ser, state):
    """Управление лазером"""
    try:
        ser.write(("LAS 1\r\n" if state else "LAS 0\r\n").encode())
    except:
        pass


def smooth_move(box, frame_w, frame_h):
    """Вычисляет новое положение серво"""
    global servo_x, servo_y

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # нормированные координаты центра
    nx, ny = cx / frame_w, cy / frame_h

    # желаемые углы
    desired_x = 90 + (nx - 0.5) * 90
    desired_y = 90 + (ny - 0.5) * 90

    # сглаживание
    desired_x = servo_x * (1 - ALPHA) + desired_x * ALPHA
    desired_y = servo_y * (1 - ALPHA) + desired_y * ALPHA

    # проверка зоны допуска
    if (abs(nx - 0.5) < DEAD_MARGIN and abs(ny - 0.5) < DEAD_MARGIN):
        return int(servo_x), int(servo_y)

    # плавное движение
    if desired_x > servo_x + STEP: servo_x += STEP
    elif desired_x < servo_x - STEP: servo_x -= STEP
    else: servo_x = desired_x

    if desired_y > servo_y + STEP: servo_y += STEP
    elif desired_y < servo_y - STEP: servo_y -= STEP
    else: servo_y = desired_y

    # ограничения
    servo_x = max(MIN_ANGLE, min(MAX_ANGLE, servo_x))
    servo_y = max(MIN_ANGLE, min(MAX_ANGLE, servo_y))

    return int(servo_x), int(servo_y)


def main():
    ser = connect_serial()
    if not ser:
        return

    # камера
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": FRAME_SIZE})
    picam2.configure(config)
    picam2.start()

    # модель
    model = YOLO(MODEL_PATH)
    print("[MODEL] YOLO загружена.")

    print("Нажмите Q для выхода.")
    while True:
        frame = picam2.capture_array()
        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)

        target_box = None
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                if int(box.cls.item()) in model.names and model.names[int(box.cls.item())] in WANTED_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    target_box = (x1, y1, x2, y2)
                    break

        if target_box:
            send_laser(ser, True)
            sx, sy = smooth_move(target_box, FRAME_SIZE[0], FRAME_SIZE[1])
            send_angles(ser, sx, sy)
            cv2.rectangle(frame, (int(target_box[0]), int(target_box[1])),
                          (int(target_box[2]), int(target_box[3])), (0, 255, 0), 2)
        else:
            send_laser(ser, False)

        cv2.imshow("AutoAim", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    send_laser(ser, False)
    ser.close()
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
