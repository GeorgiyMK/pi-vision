#!/usr/bin/env python3
import cv2
import time
import serial
import serial.tools.list_ports
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Настройки ---
MODEL_PATH = "yolov8n.pt"
WANTED_CLASSES = ["cup"]     # какие классы отслеживаем
CONF_THRESHOLD = 0.4

FRAME_SIZE = (640, 480)
ROTATE_DEG = 180

# диапазон сервоприводов
MIN_ANGLE = 30
MAX_ANGLE = 150

# параметры сглаживания
ALPHA = 0.3     # коэффициент фильтра (0..1) — выше = быстрее
STEP = 3        # максимальный шаг за один кадр (градусы)

# --- глобальные переменные ---
servo_x, servo_y = 90, 90
ser = None


# --- подключение к Arduino ---
def connect_serial():
    global ser
    try:
        port = next((p.device for p in serial.tools.list_ports.comports()
                     if "USB" in p.device or "ACM" in p.device), None)
        if not port:
            print("[SERIAL] Arduino не найдена")
            return False
        ser = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(2)
        print(f"[SERIAL] Подключено: {port}")
        return True
    except Exception as e:
        print(f"[SERIAL] Ошибка подключения: {e}")
        return False


def send_angles(x, y):
    global ser
    if ser and ser.is_open:
        cmd = f"X:{int(x)},Y:{int(y)}\r\n"
        ser.write(cmd.encode())


def send_laser(on):
    global ser
    if ser and ser.is_open:
        ser.write(b"LAS 1\r\n" if on else b"LAS 0\r\n")


# --- управление серво ---
def smooth_move(box, frame_w, frame_h):
    global servo_x, servo_y

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # нормированные координаты центра (0..1)
    nx, ny = cx / frame_w, cy / frame_h

    # преобразование в угол серво
    desired_x = MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * nx
    desired_y = MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * ny

    # сглаживание
    desired_x = servo_x * (1 - ALPHA) + desired_x * ALPHA
    desired_y = servo_y * (1 - ALPHA) + desired_y * ALPHA

    # зона допуска: если лазер внутри объекта → не двигать
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        return int(servo_x), int(servo_y)

    # плавное движение по шагам
    if desired_x > servo_x + STEP:
        servo_x += STEP
    elif desired_x < servo_x - STEP:
        servo_x -= STEP
    else:
        servo_x = desired_x

    if desired_y > servo_y + STEP:
        servo_y += STEP
    elif desired_y < servo_y - STEP:
        servo_y -= STEP
    else:
        servo_y = desired_y

    # ограничение углов
    servo_x = max(MIN_ANGLE, min(MAX_ANGLE, servo_x))
    servo_y = max(MIN_ANGLE, min(MAX_ANGLE, servo_y))

    return int(servo_x), int(servo_y)


# --- основной цикл ---
def main():
    global servo_x, servo_y

    if not connect_serial():
        return

    # камера
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE})
    picam2.configure(config)
    picam2.start()

    # модель YOLO
    model = YOLO(MODEL_PATH)
    name2id = {v: k for k, v in model.names.items()}
    wanted_ids = [name2id[n] for n in WANTED_CLASSES if n in name2id]

    print("[INFO] Система запущена. Нажмите Q для выхода.")
    send_laser(False)

    while True:
        frame = picam2.capture_array()
        if ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        target_box = None
        if boxes:
            max_area = 0
            for box in boxes:
                if int(box.cls.item()) in wanted_ids:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        target_box = (x1, y1, x2, y2)

        if target_box:
            send_laser(True)
            servo_x, servo_y = smooth_move(target_box, FRAME_SIZE[0], FRAME_SIZE[1])
            send_angles(servo_x, servo_y)

            # отладка
            cx = (target_box[0] + target_box[2]) / 2
            cy = (target_box[1] + target_box[3]) / 2
            print(f"[DEBUG] obj=({int(cx)},{int(cy)}) → servo=({servo_x},{servo_y})")

            # рисуем прямоугольник объекта
            x1, y1, x2, y2 = map(int, target_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            send_laser(False)

        cv2.imshow("Auto Aim", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    send_laser(False)
    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
