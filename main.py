#!/usr/bin/env python3
import cv2
import time
import json
import os
import numpy as np
import serial
import serial.tools.list_ports
from picamera2 import Picamera2
from ultralytics import YOLO


class Settings:
    """Класс для хранения всех настроек и констант."""
    # --- Настройки камеры и модели ---
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]  # Объекты для отслеживания
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180

    # --- Параметры поля зрения камеры (Field of View) ---
    # Для Raspberry Pi Camera Module v3 значения примерно 66 (H) и 41 (V)
    FOV_H_DEG = 66.0
    FOV_V_DEG = 41.0

    # --- Настройки управления сервоприводами ---
    MIN_ANGLE = 30
    MAX_ANGLE = 150
    DEADBAND_PX = 4  # "Мертвая зона" в пикселях, в которой ошибка считается нулевой
    MAX_STEP_DEG = 5.0  # Максимальное изменение угла за один цикл

    # --- Настройки PID-регулятора ---
    KP = 0.4  # НАЧАЛЬНЫЙ Пропорциональный коэффициент (скорость реакции)
    KI = 0.03  # Интегральный коэффициент (коррекция статической ошибки)
    KD = 0.0  # Дифференциальный коэффициент (не используется)
    SMOOTH_FACTOR = 0.6  # Коэффициент сглаживания ошибки
    BIAS_LIMIT = 15.0  # Максимальное накопленное смещение (bias)

    # --- Настройки состояний ---
    MISS_FRAMES_TO_SEARCH = 12  # Кадров без точки лазера до перехода в режим SEARCH
    SEARCH_RADIUS_STEP = 2.5  # Шаг увеличения радиуса спирали (градусы)
    SEARCH_ANGLE_STEP = 18  # Шаг поворота спирали (градусы)

    # --- Настройки Serial ---
    BAUDRATE = 115200
    RECONNECT_DELAY_S = 2.0

    # --- Файлы ---
    CALIB_PATH = "calib.json"


class PIDController:
    """Простой PID-регулятор для управления сервоприводами."""

    def __init__(self, Ki, Kd, smooth_factor, bias_limit):
        # Kp теперь управляется извне
        self.Ki = Ki
        self.Kd = Kd
        self.smooth_factor = smooth_factor
        self.bias_limit = bias_limit
        self.bias = 0.0
        self.smoothed_error = 0.0
        self.last_error = 0.0

    def calculate_step(self, error, Kp):
        """Рассчитывает шаг для сервопривода на основе ошибки."""
        self.smoothed_error = self.smooth_factor * self.smoothed_error + (1 - self.smooth_factor) * error
        self.bias = clamp(self.bias + self.Ki * self.smoothed_error, -self.bias_limit, self.bias_limit)

        # В этой версии мы используем только P и I часть для стабильности
        # step = Kp * self.smoothed_error + self.bias

        # УПРОЩЕННАЯ ВЕРСИЯ ДЛЯ НАСТРОЙКИ: ТОЛЬКО P-КОНТРОЛЛЕР
        step = Kp * error

        return step

    def reset(self):
        """Сбрасывает состояние регулятора."""
        self.bias = 0.0
        self.smoothed_error = 0.0
        self.last_error = 0.0


class StateMachine:
    """Управляет состояниями системы (IDLE, CHASE, TRACK, SEARCH)."""

    def __init__(self):
        self.state = "IDLE"
        self.miss_dot_counter = 0

    def update(self, obj_detected, dot_detected):
        """Обновляет состояние на основе наличия объекта и точки лазера."""
        prev_state = self.state

        if obj_detected and dot_detected:
            self.state = "TRACK"
            self.miss_dot_counter = 0
        elif obj_detected and not dot_detected:
            self.miss_dot_counter += 1
            if self.miss_dot_counter >= Settings.MISS_FRAMES_TO_SEARCH:
                self.state = "SEARCH"
            else:
                self.state = "CHASE"
        else:
            self.state = "IDLE"
            self.miss_dot_counter = 0

        if self.state != prev_state:
            print(f"[STATE] {prev_state} -> {self.state}")

        return self.state


class TurretController:
    """Основной класс, управляющий всей системой."""

    def __init__(self):
        self.settings = Settings()
        self.picam2 = self._init_camera()
        self.model = self._init_model()
        self.ser = None
        self.connect_serial()

        self.w, self.h = self.settings.FRAME_SIZE
        self.center_x, self.center_y = self.w // 2, self.h // 2

        self.servo_x, self.servo_y = 90.0, 90.0
        self.laser_on = False

        # Динамический Kp для настройки
        self.kp = self.settings.KP

        self.pid_x = PIDController(self.settings.KI, self.settings.KD, self.settings.SMOOTH_FACTOR,
                                   self.settings.BIAS_LIMIT)
        self.pid_y = PIDController(self.settings.KI, self.settings.KD, self.settings.SMOOTH_FACTOR,
                                   self.settings.BIAS_LIMIT)

        self.state_machine = StateMachine()
        self.search_radius = 0.0
        self.search_angle = 0.0

        self.calib = self._load_calib()
        self._apply_calib()

    def _init_camera(self):
        """Инициализация камеры."""
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": self.settings.FRAME_SIZE})
            picam2.configure(config)
            picam2.start()
            print("[CAMERA] Камера успешно инициализирована.")
            return picam2
        except Exception as e:
            print(f"[CAMERA] Ошибка инициализации камеры: {e}")
            exit()

    def _init_model(self):
        """Инициализация модели YOLO."""
        try:
            model = YOLO(self.settings.MODEL_PATH)
            self.name2id = {v: k for k, v in model.names.items()}
            self.wanted_ids = [self.name2id[n] for n in self.settings.WANTED_CLASSES if n in self.name2id]
            print(f"[MODEL] Модель YOLOv8 загружена. Отслеживаемые классы: {self.settings.WANTED_CLASSES}")
            return model
        except Exception as e:
            print(f"[MODEL] Ошибка загрузки модели YOLO: {e}")
            exit()

    def _load_calib(self):
        """Загрузка файла калибровки."""
        if os.path.exists(self.settings.CALIB_PATH):
            try:
                with open(self.settings.CALIB_PATH, "r") as f:
                    cal_data = json.load(f)
                    print(f"[CALIB] Калибровка загружена: {cal_data}")
                    return cal_data
            except Exception as e:
                print(f"[CALIB] Ошибка загрузки калибровки: {e}")
        # Возвращаем словарь с настройками по умолчанию, если файл не найден или поврежден
        return {"invert_x": False, "invert_y": False}

    def _save_calib(self):
        """Сохранение файла калибровки."""
        self.calib = {
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
        }
        try:
            with open(self.settings.CALIB_PATH, "w") as f:
                json.dump(self.calib, f, indent=2)
            print(f"[CALIB] Калибровка сохранена: {self.calib}")
        except Exception as e:
            print(f"[CALIB] Ошибка сохранения калибровки: {e}")

    def _apply_calib(self):
        """Применение загруженных настроек калибровки."""
        self.invert_x = self.calib.get("invert_x", False)
        self.invert_y = self.calib.get("invert_y", False)

    def connect_serial(self):
        """Поиск и подключение к Arduino."""
        if self.ser and self.ser.is_open:
            return True
        try:
            port = next(
                (p.device for p in serial.tools.list_ports.comports() if "USB" in p.device or "ACM" in p.device), None)
            if not port:
                if self.ser: print("[SERIAL] Arduino не найдена.")
                self.ser = None
                return False
            self.ser = serial.Serial(port, self.settings.BAUDRATE, timeout=0.1)
            time.sleep(1.8)  # Даем Arduino время на перезагрузку
            print(f"[SERIAL] Подключено к Arduino: {port}")
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Ошибка подключения: {e}")
            self.ser = None
            return False

    def _send_command(self, command):
        """Отправка команды на Arduino с проверкой подключения."""
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                return
        try:
            self.ser.write(command.encode())
        except serial.SerialException as e:
            print(f"[SERIAL] Ошибка отправки команды: {e}. Закрытие порта.")
            if self.ser: self.ser.close()
            self.ser = None

    def send_angles(self, x, y):
        self._send_command(f"X:{int(x)},Y:{int(y)}\r\n")

    def send_laser(self, on):
        self.laser_on = on
        self._send_command("LAS 1\r\n" if on else "LAS 0\r\n")

    def run(self):
        """Основной цикл программы."""
        print("Нажмите 'Q' в окне для выхода.")
        print("Клавиши: L-лазер, C-центр, I/K-инверт, S-сохранить, U/J-скорость")
        try:
            while True:
                frame = self._capture_and_process_frame()

                results = self.model.predict(frame, conf=self.settings.CONF_THRESHOLD, imgsz=self.settings.IMGSZ,
                                             verbose=False)
                target_pos = self._find_largest_target(results)
                laser_pos = self._detect_laser_dot(frame)

                current_state = self.state_machine.update(target_pos is not None, laser_pos is not None)

                self._handle_state(current_state, target_pos, laser_pos)

                self._draw_overlay(frame, target_pos, laser_pos)
                cv2.imshow("Auto Aim", frame)

                if self._handle_keyboard():
                    break

        finally:
            self.cleanup()

    def _capture_and_process_frame(self):
        """Захват и поворот кадра."""
        frame = self.picam2.capture_array()
        if self.settings.ROTATE_DEG == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _find_largest_target(self, results):
        """Находит самый большой объект из заданных классов."""
        if not results or len(results[0].boxes) == 0: return None
        boxes = results[0].boxes
        largest_target = None;
        max_area = 0
        for box in boxes:
            if int(box.cls.item()) in self.wanted_ids:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    cx = int((x1 + x2) / 2);
                    cy = int((y1 + y2) / 2)
                    largest_target = (cx, cy)
        return largest_target

    def _detect_laser_dot(self, frame):
        """Обнаруживает красную точку лазера."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 80, 180]);
        upper1 = np.array([12, 255, 255])
        lower2 = np.array([168, 80, 180]);
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1);
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 2: return None
        M = cv2.moments(c)
        if M["m00"] == 0: return None
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def _handle_state(self, state, target_pos, laser_pos):
        """Выполняет действия в зависимости от текущего состояния."""
        if state == "IDLE":
            if self.laser_on: self.send_laser(False)
            self.servo_x += (90 - self.servo_x) * 0.1
            self.servo_y += (90 - self.servo_y) * 0.1
            self.send_angles(self.servo_x, self.servo_y)
            self.search_radius = 0.0
            return

        if not self.laser_on: self.send_laser(True)

        if state == "TRACK":
            error_px_x = target_pos[0] - laser_pos[0]
            error_px_y = target_pos[1] - laser_pos[1]
            self._update_servos_from_pixel_error(error_px_x, error_px_y)
        elif state == "CHASE":
            error_px_x = target_pos[0] - self.center_x
            error_px_y = target_pos[1] - self.center_y
            self._update_servos_from_pixel_error(error_px_x, error_px_y)
        elif state == "SEARCH":
            self.search_radius += self.settings.SEARCH_RADIUS_STEP / 10
            self.search_angle = (self.search_angle + self.settings.SEARCH_ANGLE_STEP) % 360
            offset_x_deg = self.search_radius * np.cos(np.deg2rad(self.search_angle))
            offset_y_deg = self.search_radius * np.sin(np.deg2rad(self.search_angle))
            target_x = 90 + (offset_x_deg if not self.invert_x else -offset_x_deg)
            target_y = 90 + (offset_y_deg if not self.invert_y else -offset_y_deg)
            self.servo_x = clamp(target_x, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
            self.servo_y = clamp(target_y, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
            self.send_angles(self.servo_x, self.servo_y)
            if self.search_radius > 25: self.search_radius = 0

    def _update_servos_from_pixel_error(self, dx_px, dy_px):
        """Обновляет положение сервоприводов на основе ошибки в пикселях."""
        if abs(dx_px) < self.settings.DEADBAND_PX: dx_px = 0
        if abs(dy_px) < self.settings.DEADBAND_PX: dy_px = 0

        error_deg_x = (dx_px / self.w) * self.settings.FOV_H_DEG
        error_deg_y = -(dy_px / self.h) * self.settings.FOV_V_DEG

        # Используем упрощенный контроллер для настройки
        step_x = self.pid_x.calculate_step(error_deg_x, self.kp)
        step_y = self.pid_y.calculate_step(error_deg_y, self.kp)

        if self.invert_x: step_x = -step_x
        if self.invert_y: step_y = -step_y

        final_step_x = clamp(step_x, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)
        final_step_y = clamp(step_y, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + final_step_x, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + final_step_y, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)

        self.send_angles(self.servo_x, self.servo_y)

    def _draw_overlay(self, frame, target_pos, laser_pos):
        """Отрисовка информации на кадре."""
        if target_pos: cv2.circle(frame, target_pos, 8, (255, 0, 0), 2)
        if laser_pos: cv2.circle(frame, laser_pos, 8, (0, 0, 255), 2)
        cv2.circle(frame, (self.center_x, self.center_y), 4, (0, 255, 0), -1)
        info_text = f"S: {self.state_machine.state} | Kp: {self.kp:.2f} | INV: {int(self.invert_x)},{int(self.invert_y)}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _handle_keyboard(self):
        """Обработка нажатий клавиш."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('l'):
            self.send_laser(not self.laser_on)
        elif key == ord('c'):
            self.servo_x, self.servo_y = 90.0, 90.0
            self.pid_x.reset()
            self.pid_y.reset()
            self.send_angles(self.servo_x, self.servo_y)
        elif key == ord('i'):
            self.invert_x = not self.invert_x
            print(f"Invert X: {self.invert_x}")
        elif key == ord('k'):
            self.invert_y = not self.invert_y
            print(f"Invert Y: {self.invert_y}")
        elif key == ord('s'):
            self._save_calib()
        elif key == ord('u'):  # Увеличить Kp
            self.kp += 0.05
            print(f"Kp set to: {self.kp:.2f}")
        elif key == ord('j'):  # Уменьшить Kp
            self.kp = max(0, self.kp - 0.05)
            print(f"Kp set to: {self.kp:.2f}")
        return False

    def cleanup(self):
        """Освобождение ресурсов перед выходом."""
        print("\nЗавершение работы...")
        if self.ser and self.ser.is_open:
            try:
                self.send_laser(False)
                for _ in range(10):
                    self.servo_x += (90 - self.servo_x) * 0.2
                    self.servo_y += (90 - self.servo_y) * 0.2
                    self.send_angles(self.servo_x, self.servo_y)
                    time.sleep(0.02)
                self.ser.close()
                print("[SERIAL] Порт закрыт.")
            except Exception as e:
                print(f"[SERIAL] Ошибка при закрытии порта: {e}")

        if self.picam2: self.picam2.stop(); print("[CAMERA] Камера остановлена.")
        cv2.destroyAllWindows()
        self._save_calib()
        print("Готово.")


def clamp(v, vmin, vmax):
    """Ограничивает значение v в диапазоне [vmin, vmax]."""
    return max(vmin, min(vmax, v))


if __name__ == '__main__':
    controller = TurretController()
    controller.run()
