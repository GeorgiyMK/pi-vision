from picamera2 import Picamera2
import cv2
import time
import numpy as np
import serial
from ultralytics import YOLO

# === ПАРАМЕТРЫ ===
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.2
ONLY = ["cup"]  # отслеживаемый объект
IMGSZ = 320
ROTATE_DEG = 180
FRAME_SIZE = (640, 480)  # уменьшено для скорости

Kp = 0.1  # чувствительность корректировки
min_angle = 30
max_angle = 150
servo_x = 90
servo_y = 90

SERIAL_PORT = "/dev/ttyACM0"  # порт Arduino
BAUDRATE = 9600

# === ИНИЦИАЛИЗАЦИЯ ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (FRAME_SIZE[0], FRAME_SIZE[1])}
))
picam2.start()

model = YOLO(MODEL_PATH)
name2id = {v: k for k, v in model.names.items()}
class_ids = [name2id[n] for n in ONLY if n in name2id]

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
time.sleep(2)

frame_center = (FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2)


def send_angles(x, y):
    ser.write(f"X:{int(x)},Y:{int(y)}\n".encode())


def detect_laser(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 200])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None


last_detect_time = time.time()
auto_hold = True

print("Нажмите 'q' для выхода")
while True:
    frame = picam2.capture_array()
    if ROTATE_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        imgsz=IMGSZ,
        classes=class_ids,
        verbose=False
    )

    boxes = results[0].boxes
    object_center = None
    if boxes and len(boxes) > 0:
        xyxy = boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        object_center = (cx, cy)
        last_detect_time = time.time()

    laser_center = detect_laser(frame)

    if object_center and laser_center:
        dx = object_center[0] - laser_center[0]
        dy = object_center[1] - laser_center[1]

        servo_x -= dx * Kp
        servo_y += dy * Kp

        servo_x = max(min(servo_x, max_angle), min_angle)
        servo_y = max(min(servo_y, max_angle), min_angle)

        send_angles(servo_x, servo_y)
        auto_hold = False

    elif time.time() - last_detect_time > 1.5 and not auto_hold:
        # Если объект потерян — возвращаем в центр
        servo_x = 90
        servo_y = 90
        send_angles(servo_x, servo_y)
        auto_hold = True

    # Отображение
    if object_center:
        cv2.circle(frame, object_center, 5, (255, 0, 0), -1)
    if laser_center:
        cv2.circle(frame, laser_center, 5, (0, 0, 255), -1)
    if object_center and laser_center:
        cv2.line(frame, object_center, laser_center, (0, 255, 255), 2)

    cv2.imshow("Auto Aim Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
