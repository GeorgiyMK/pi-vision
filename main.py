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


# =====================================================================================
# КЛАССЫ КОНФИГУРАЦИИ И УПРАВЛЕНИЯ
# =====================================================================================

class Settings:
    """Класс для хранения всех настроек и констант."""
    # --- Модель и камера ---
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180

    # --- Механика и управление ---
    MIN_ANGLE = 30
    MAX_ANGLE = 150
    DEADBAND_PX = 3
    MAX_STEP_DEG = 7.0

    # --- PID-регулятор ---
    KP = 0.8  # Скорость реакции (пропорциональный)
    KI = 0.04  # Устранение статической ошибки (интегральный)
    KD = 0.1  # Демпфирование колебаний (дифференциальный)
    BIAS_LIMIT = 20.0

    # --- Состояния ---
    MISS_FRAMES_TO_SEARCH = 15
    SEARCH_RADIUS_MAX_DEG = 25.0
    SEARCH_SPEED_FACTOR = 0.1

    # --- Serial и файлы ---
    BAUDRATE = 115200
    CALIB_PATH = "calib.json"


class PIDController:
    """PID-регулятор для плавного и точного управления."""

    def __init__(self, Kp, Ki, Kd, bias_limit):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.bias_limit = bias_limit
        self.bias = 0.0  # Интегральная сумма
        self.last_error = 0.0

    def calculate_step(self, error):
        self.bias = clamp(self.bias + error * self.Ki, -self.bias_limit, self.bias_limit)
        derivative = error - self.last_error
        self.last_error = error
        return self.Kp * error + self.bias + self.Kd * derivative

    def reset(self):
        self.bias = 0.0
        self.last_error = 0.0


class StateMachine:
    """Управляет состояниями системы, включая калибровку."""
    VALID_STATES = ["IDLE", "CHASE", "TRACK", "SEARCH", "CALIBRATING"]

    def __init__(self):
        self.state = "IDLE"
        self.miss_dot_counter = 0

    def set_state(self, new_state):
        if new_state not in self.VALID_STATES: return
        if self.state != new_state:
            print(f"[STATE] {self.state} -> {new_state}")
            self.state = new_state
            if new_state != "CHASE": self.miss_dot_counter = 0

    def update(self, obj_detected, dot_detected):
        if self.state == "CALIBRATING": return  # В режиме калибровки состояние не меняется

        if obj_detected and dot_detected:
            self.set_state("TRACK")
        elif obj_detected and not dot_detected:
            self.miss_dot_counter += 1
            if self.miss_dot_counter >= Settings.MISS_FRAMES_TO_SEARCH:
                self.set_state("SEARCH")
            else:
                self.set_state("CHASE")
        else:
            self.set_state("IDLE")


# =====================================================================================
# КЛАСС АВТОМАТИЧЕСКОЙ КАЛИБРОВКИ
# =====================================================================================

class AutoCalibrator:
    """Выполняет процедуру автоматической калибровки системы."""

    def __init__(self, controller):
        self.ctrl = controller
        self.step = "IDLE"
        self.cal_points = {
            "p1_angle": (Settings.MIN_ANGLE + 15, Settings.MIN_ANGLE + 15),
            "p2_angle": (Settings.MAX_ANGLE - 15, Settings.MAX_ANGLE - 15)
        }
        self.measurements = {}
        self.start_time = 0

    def start(self):
        print("[CALIB] Начало автоматической калибровки.")
        self.ctrl.state_machine.set_state("CALIBRATING")
        self.step = "GOTO_P1"
        self.start_time = time.time()
        self.ctrl.send_laser(True)
        self.ctrl.send_angles(*self.cal_points["p1_angle"])
        return True

    def run_step(self, laser_pos):
        """Выполняет один шаг калибровочной машины состояний."""
        if self.step == "IDLE": return None

        # Ожидание стабилизации сервоприводов
        if time.time() - self.start_time < 2.0: return None

        if self.step == "GOTO_P1":
            if not laser_pos: self._fail("Лазер не найден в точке 1"); return None
            self.measurements["p1_pixel"] = laser_pos
            print(f"[CALIB] Точка 1 замерена: Угол={self.cal_points['p1_angle']}, Пиксель={laser_pos}")
            self.step = "GOTO_P2"
            self.start_time = time.time()
            self.ctrl.send_angles(*self.cal_points["p2_angle"])

        elif self.step == "GOTO_P2":
            if not laser_pos: self._fail("Лазер не найден в точке 2"); return None
            self.measurements["p2_pixel"] = laser_pos
            print(f"[CALIB] Точка 2 замерена: Угол={self.cal_points['p2_angle']}, Пиксель={laser_pos}")
            self._calculate_results()

        return None  # Процесс не завершен

    def _calculate_results(self):
        """Вычисляет параметры калибровки на основе двух точек."""
        p1_ax, p1_ay = self.cal_points["p1_angle"]
        p2_ax, p2_ay = self.cal_points["p2_angle"]
        p1_px, p1_py = self.measurements["p1_pixel"]
        p2_px, p2_py = self.measurements["p2_pixel"]

        delta_angle_x = p2_ax - p1_ax
        delta_pixel_x = p2_px - p1_px
        if abs(delta_angle_x) < 1 or abs(delta_pixel_x) < 10: self._fail("Недостаточное смещение по оси X"); return

        delta_angle_y = p2_ay - p1_ay
        delta_pixel_y = p2_py - p1_py
        if abs(delta_angle_y) < 1 or abs(delta_pixel_y) < 10: self._fail("Недостаточное смещение по оси Y"); return

        px_per_deg_x = delta_pixel_x / delta_angle_x
        px_per_deg_y = delta_pixel_y / delta_angle_y

        center_offset_px_x = p1_px - (p1_ax - 90.0) * px_per_deg_x
        center_offset_px_y = p1_py - (p1_ay - 90.0) * px_per_deg_y

        new_calib = {
            "px_per_deg_x": px_per_deg_x,
            "px_per_deg_y": px_per_deg_y,
            "center_offset_px_x": center_offset_px_x,
            "center_offset_px_y": center_offset_px_y,
        }

        print("[CALIB] Калибровка успешно завершена!")
        print(f"  - Масштаб X: {px_per_deg_x:.2f} px/deg")
        print(f"  - Масштаб Y: {px_per_deg_y:.2f} px/deg")
        print(f"  - Смещение центра: ({center_offset_px_x:.1f}, {center_offset_px_y:.1f}) px")

        self.step = "IDLE"
        self.ctrl.apply_and_save_calib(new_calib)
        self.ctrl.state_machine.set_state("IDLE")

    def _fail(self, reason):
        print(f"[CALIB] ОШИБКА КАЛИБРОВКИ: {reason}")
        self.step = "IDLE"
        self.ctrl.state_machine.set_state("IDLE")


# =====================================================================================
# ОСНОВНОЙ КЛАСС КОНТРОЛЛЕРА
# =====================================================================================

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

        self.pid_x = PIDController(self.settings.KP, self.settings.KI, self.settings.KD, self.settings.BIAS_LIMIT)
        self.pid_y = PIDController(self.settings.KP, self.settings.KI, self.settings.KD, self.settings.BIAS_LIMIT)

        self.state_machine = StateMachine()
        self.calibrator = AutoCalibrator(self)

        self.search_angle = 0.0

        self.calib = self._load_calib()

    # --- Методы инициализации и связи ---
    def _init_camera(self):
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"format": "RGB888", "size": self.settings.FRAME_SIZE})
            picam2.configure(config)
            picam2.start();
            print("[INIT] Камера запущена.")
            return picam2
        except Exception as e:
            print(f"[FATAL] Ошибка камеры: {e}"); exit()

    def _init_model(self):
        try:
            model = YOLO(self.settings.MODEL_PATH)
            self.wanted_ids = [k for k, v in model.names.items() if v in self.settings.WANTED_CLASSES]
            print(f"[INIT] Модель YOLO загружена.")
            return model
        except Exception as e:
            print(f"[FATAL] Ошибка модели: {e}"); exit()

    def connect_serial(self):
        if self.ser and self.ser.is_open: return True
        try:
            port = next(
                (p.device for p in serial.tools.list_ports.comports() if "USB" in p.device or "ACM" in p.device), None)
            if not port: self.ser = None; return False
            self.ser = serial.Serial(port, self.settings.BAUDRATE, timeout=0.1)
            time.sleep(1.8)
            print(f"[SERIAL] Подключено к Arduino: {port}")
            return True
        except serial.SerialException as e:
            self.ser = None; return False

    def send_command(self, command):
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial(): return
        try:
            self.ser.write(command.encode())
        except serial.SerialException:
            self.ser.close(); self.ser = None

    def send_angles(self, x, y):
        # ИЗМЕНЕНО: Отправка команды в формате, который понимает ваш скетч Arduino
        self.send_command(f"SET {int(x)} {int(y)}\r\n")

    def send_laser(self, on):
        self.laser_on = on
        # ИЗМЕНЕНО: Отправка команды в формате, который понимает ваш скетч Arduino
        self.send_command("LASER 1\r\n" if on else "LASER 0\r\n")

    # --- Методы калибровки ---
    def _load_calib(self):
        """Загружает калибровку или возвращает None, если она невалидна."""
        try:
            if os.path.exists(self.settings.CALIB_PATH):
                with open(self.settings.CALIB_PATH, "r") as f:
                    c = json.load(f)
                if all(k in c for k in ["px_per_deg_x", "px_per_deg_y", "center_offset_px_x", "center_offset_px_y"]):
                    print("[CALIB] Рабочая калибровка загружена.")
                    return c
        except Exception as e:
            print(f"[CALIB] Ошибка чтения файла калибровки: {e}")

        print("[WARN] Калибровка не найдена или повреждена. Требуется автоматическая калибровка (клавиша 'A').")
        return None

    def apply_and_save_calib(self, calib_data):
        self.calib = calib_data
        try:
            with open(self.settings.CALIB_PATH, "w") as f:
                json.dump(self.calib, f, indent=2)
            print("[CALIB] Новая калибровка сохранена.")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить калибровку: {e}")

    # --- Основной цикл и логика состояний ---
    def run(self):
        print("\nСистема наведения готова. Клавиши: [A] - Автокалибровка, [Q] - Выход.")
        try:
            while True:
                frame = self.picam2.capture_array()
                if self.settings.ROTATE_DEG == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)

                results = self.model.predict(frame, conf=self.settings.CONF_THRESHOLD, imgsz=self.settings.IMGSZ,
                                             verbose=False)
                target_pos = self._find_largest_target(results)
                laser_pos = self._detect_laser_dot(frame)

                if self.state_machine.state == "CALIBRATING":
                    self.calibrator.run_step(laser_pos)
                else:
                    self.state_machine.update(target_pos is not None, laser_pos is not None)
                    self._handle_state(target_pos, laser_pos)

                self._draw_overlay(frame, target_pos, laser_pos)
                cv2.imshow("Guidance System", frame)

                if self._handle_keyboard(): break
        finally:
            self.cleanup()

    def _handle_state(self, target_pos, laser_pos):
        """Выполняет действия в зависимости от текущего состояния."""
        if not self.calib and self.state_machine.state != "IDLE":
            self.state_machine.set_state("IDLE")

        state = self.state_machine.state
        if state == "IDLE":
            if self.laser_on: self.send_laser(False)
            self.send_command("HOME\r\n")  # Используем HOME команду из вашего скетча
            # Обновляем внутренние переменные углов
            self.servo_x += (90 - self.servo_x) * 0.1
            self.servo_y += (90 - self.servo_y) * 0.1
            return

        if not self.laser_on: self.send_laser(True)

        if state in ["TRACK", "CHASE"]:
            current_pos = laser_pos if state == "TRACK" else \
                (self.calib["center_offset_px_x"], self.calib["center_offset_px_y"])

            error_px_x = target_pos[0] - current_pos[0]
            error_px_y = target_pos[1] - current_pos[1]
            self._update_servos_from_pixel_error(error_px_x, error_px_y)

        elif state == "SEARCH":
            self.search_angle = (self.search_angle + 15) % 360
            radius_factor = abs(((self.state_machine.miss_dot_counter / self.settings.MISS_FRAMES_TO_SEARCH) * 2) - 1)
            radius = self.settings.SEARCH_RADIUS_MAX_DEG * (1 - radius_factor)

            offset_x = radius * np.cos(np.deg2rad(self.search_angle))
            offset_y = radius * np.sin(np.deg2rad(self.search_angle))

            offset_px_x = offset_x * self.calib["px_per_deg_x"]
            offset_px_y = offset_y * self.calib["px_per_deg_y"]

            error_px_x = (target_pos[0] + offset_px_x) - self.calib["center_offset_px_x"]
            error_px_y = (target_pos[1] + offset_px_y) - self.calib["center_offset_px_y"]
            self._update_servos_from_pixel_error(error_px_x, error_px_y)

    def _update_servos_from_pixel_error(self, dx_px, dy_px):
        """Ключевой метод: преобразует ошибку в пикселях в команды для серво."""
        if not self.calib: return
        if abs(dx_px) < self.settings.DEADBAND_PX: dx_px = 0
        if abs(dy_px) < self.settings.DEADBAND_PX: dy_px = 0

        error_deg_x = dx_px / self.calib["px_per_deg_x"]
        error_deg_y = dy_px / self.calib["px_per_deg_y"]

        step_x = self.pid_x.calculate_step(error_deg_x)
        step_y = self.pid_y.calculate_step(error_deg_y)

        final_step_x = clamp(step_x, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)
        final_step_y = clamp(step_y, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + final_step_x, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + final_step_y, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)

        self.send_angles(self.servo_x, self.servo_y)

    # --- Вспомогательные методы ---
    def _find_largest_target(self, results):
        if not results or len(results[0].boxes) == 0: return None
        largest_target = None;
        max_area = 0
        for box in results[0].boxes:
            if int(box.cls.item()) in self.wanted_ids:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_target = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return largest_target

    def _detect_laser_dot(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 90, 190), (10, 255, 255)) + \
               cv2.inRange(hsv, (170, 90, 190), (180, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 2: return None
        M = cv2.moments(c)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] > 0 else None

    def _draw_overlay(self, frame, target_pos, laser_pos):
        if self.calib:
            cx, cy = int(self.calib['center_offset_px_x']), int(self.calib['center_offset_px_y'])
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
        if target_pos: cv2.circle(frame, target_pos, 10, (255, 0, 0), 2)
        if laser_pos: cv2.circle(frame, laser_pos, 10, (0, 0, 255), 2)

        state_text = self.state_machine.state
        if self.state_machine.state == "CALIBRATING": state_text += f" ({self.calibrator.step})"
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not self.calib:
            cv2.putText(frame, "ТРЕБУЕТСЯ КАЛИБРОВКА (НАЖМИТЕ 'A')", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2)

    def _handle_keyboard(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('a'):
            if self.state_machine.state != "CALIBRATING": self.calibrator.start()
        return False

    def cleanup(self):
        print("\nЗавершение работы...")
        if self.ser and self.ser.is_open:
            self.send_laser(False)
            self.send_command("HOME\r\n")
            self.ser.close();
            print("[CLEANUP] Serial порт закрыт.")
        if self.picam2: self.picam2.stop(); print("[CLEANUP] Камера остановлена.")
        cv2.destroyAllWindows()
        print("Готово.")


def clamp(v, vmin, vmax): return max(vmin, min(vmax, v))


# =====================================================================================
# ТОЧКА ВХОДА
# =====================================================================================

if __name__ == '__main__':
    controller = TurretController()
    controller.run()

