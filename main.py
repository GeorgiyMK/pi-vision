from picamera2 import Picamera2
import cv2, time
from ultralytics import YOLO

# --- настройки ---
MODEL_PATH = "yolov8n.pt"   # базовая модель на COCO (80 классов)
CONF_THRES = 0.3            # порог уверенности
ROTATE_DEG = 180              # 0/90/180/270 — если нужно вверх ногами, поставьте 180
ONLY = ["cup"]                   # например: ["person","cat","dog"] — чтобы оставить только нужные классы
IMGSZ = 640                 # размер входа модели (320/480/640) — меньше = быстрее
# -----------------

# инициализация камеры
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (1280, 720)}  # можно снизить до (960,540) или (640,480) для скорости
))
picam2.start()

# загрузка модели
model = YOLO(MODEL_PATH)

# вычислим индексы классов, если заданы имена
class_ids = None
if ONLY:
    name2id = {v:k for k,v in model.names.items()}
    class_ids = [name2id[n] for n in ONLY if n in name2id]

print("Нажмите 'q' для выхода, 's' чтобы сохранить кадр.")
while True:
    frame = picam2.capture_array()

    # поворот, если требуется
    if ROTATE_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif ROTATE_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATE_DEG == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # инференс; classes=... оставит только указанные классы
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        imgsz=IMGSZ,
        classes=class_ids,
        verbose=False
    )

    # готовая отрисовка боксов
    annotated = results[0].plot()

    cv2.imshow("Pi Vision (YOLOv8n)", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"det_{int(time.time())}.jpg", annotated)
        print("Сохранён кадр.")

cv2.destroyAllWindows()
picam2.stop()
