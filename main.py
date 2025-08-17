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
    """Все настройки и константы."""

    # --- Камера / модель ---
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]         # какие классы отслеживаем
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180                 # 0/90/180/270

    # --- Поле зрения камеры (CM3, обычная линза ≈) ---
    FOV_H_DEG = 66.0
    FOV_V_DEG = 41.0

    # --- Управление сервами ---
    MIN_ANGLE = 30
    MAX_ANGLE = 150
    DEADBAND_PX = 10                 # было 4 — сделаем больше, чтобы не «рыскал»
    MAX_STEP_DEG = 2.5               # было 5.0 — мягче шаг

    # --- PID (теперь PI: P + I) ---
    KP = 0.18                        # стартовое значение (0.15–0.25)
    KI = 0.015                       # медленная подтяжка остаточной ошибки
    KD = 0.0
    SMOOTH_FACTOR = 0.6              # сглаживание ошибки
    BIAS_LIMIT = 15.0

    # --- Состояния / гистерезис ---
    MISS_FRAMES_TO_SEARCH = 45       # было 12 — меньше дребезга состояний

    # --- Поиск цели (спираль в SEARCH) ---
    SEARCH_RADIUS_STEP = 2.5
    SEARCH_ANGLE_STEP = 18

    # --- Serial ---
    BAUDRATE = 115200
    RECONNECT_DELAY_S = 2.0
    CMD_HZ = 25.0                    # лимит частоты отправки углов (Гц)

    # --- Калибровка ---
    CALIB_PATH = "calib.json"


def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))


class PIDController:
    """PI-регулятор (P + I)."""

    def __init__(self, Ki, Kd, smooth_factor, bias_limit):
        self.Ki = Ki
        self.Kd = Kd
        self.smooth_factor = smooth_factor
        self.bias_limit = bias_limit
        self.bias = 0.0
        self.smoothed_error = 0.0
        self.last_error = 0.0

    def calculate_step(self, error, Kp):
        # Сглаживаем ошибку, копим интегральную составляющую в пределах
        self.smoothed_error = (
            self.smooth_factor * self.smoothed_error
            + (1 - self.smooth_factor) * error
        )
        self.bias = clamp(self.bias + self.Ki * self.smoothed_error,
                          -self.bias_limit, self.bias_limit)

        # PI: пропорциональная + интегральная часть
        step = Kp * self.smoothed_error + self.bias
        return step

    def reset(self):
        self.bias = 0.0
        self.smoothed_error = 0.0
        self.last_error = 0.0


class StateMachine:
    """IDLE / CHASE / TRACK / SEARCH."""

    def __init__(self):
        self.state = "IDLE"
        self.miss_dot_counter = 0

    def update(self, obj_detected, dot_detected):
        prev = self.state
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

        if self.state != prev:
            print(f"[STATE] {prev} -> {self.state}")
        return self.state


class TurretController:
    """Главный класс управления."""

    def __init__(self):
        self.settings = Settings()
        self.picam2 = self._init_camera()
        self.model = self._init_model()
        self.ser = None
        self.connect_serial()

        self.w, self.h = self.settings.FRAME_SIZE
        self.center_x, self.center_y = self.w // 2, self.h // 2

        # текущие углы (0..180)
        self.servo_x, self.servo_y = 90.0, 90.0
        self.laser_on = False

        # настраиваемый Kp
        self.kp = self.settings.KP

        self.pid_x = PIDController(self.settings.KI, self.settings.KD,
                                   self.settings.SMOOTH_FACTOR, self.settings.BIAS_LIMIT)
        self.pid_y = PIDController(self.settings.KI, self.settings.KD,
                                   self.settings.SMOOTH_FACTOR, self.settings.BIAS_LIMIT)

        self.state_machine = StateMachine()
        self.search_radius = 0.0
        self.search_angle = 0.0

        # Память цели и точка лазера для гистерезиса
        self.prev_target = None
        self.prev_target_time = 0.0
        self.last_laser_seen = 0.0

        # лимит частоты отправки углов
        self._next_tx = 0.0

        # калибровка инверсий
        self.calib = self._load_calib()
        self._apply_calib()

    # ---------- Инициализация ----------

    def _init_camera(self):
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"format": "RGB888", "size": self.settings.FRAME_SIZE}
            )
            picam2.configure(config)
            picam2.start()
            print("[CAMERA] ok")
            return picam2
        except Exception as e:
            print(f"[CAMERA] Ошибка: {e}")
            raise

    def _init_model(self):
        try:
            model = YOLO(self.settings.MODEL_PATH)
            self.name2id = {v: k for k, v in model.names.items()}
            self.wanted_ids = [self.name2id[n]
                               for n in self.settings.WANTED_CLASSES
                               if n in self.name2id]
            print(f"[MODEL] Загружено. Классы: {self.settings.WANTED_CLASSES}")
            return model
        except Exception as e:
            print(f"[MODEL] Ошибка: {e}")
            raise

    def connect_serial(self):
        """Автопоиск /dev/ttyUSB*|ttyACM* и подключение."""
        if self.ser and self.ser.is_open:
            return True
        try:
            port = next((p.device for p in serial.tools.list_ports.comports()
                         if "/ttyUSB" in p.device or "/ttyACM" in p.device), None)
            if not port:
                print("[SERIAL] Arduino не найдена.")
                self.ser = None
                return False
            self.ser = serial.Serial(port, self.settings.BAUDRATE, timeout=0.1)
            time.sleep(1.8)  # ждём автоперезапуск Arduino
            print(f"[SERIAL] Подключено: {port}")
            # необязательный хендшейк:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b"PING\r\n")
                time.sleep(0.2)
                data = self.ser.read(100).decode(errors="ignore")
                if data:
                    print("[SERIAL] RX:", data.strip())
            except Exception:
                pass
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Ошибка: {e}")
            self.ser = None
            return False

    # ---------- Калибровка ----------

    def _load_calib(self):
        if os.path.exists(self.settings.CALIB_PATH):
            try:
                with open(self.settings.CALIB_PATH, "r") as f:
                    c = json.load(f)
                    print(f"[CALIB] Загружено: {c}")
                    return c
            except Exception as e:
                print(f"[CALIB] Ошибка загрузки: {e}")
        return {"invert_x": False, "invert_y": False}

    def _save_calib(self):
        self.calib = {"invert_x": self.invert_x, "invert_y": self.invert_y}
        try:
            with open(self.settings.CALIB_PATH, "w") as f:
                json.dump(self.calib, f, indent=2)
            print(f"[CALIB] Сохранено: {self.calib}")
        except Exception as e:
            print(f"[CALIB] Ошибка сохранения: {e}")

    def _apply_calib(self):
        self.invert_x = self.calib.get("invert_x", False)
        self.invert_y = self.calib.get("invert_y", False)

    # ---------- Коммуникация ----------

    def _send_command(self, line: str):
        """Отправка строки в Serial (CRLF). Автовосстановление при разрыве."""
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                return
        try:
            if not line.endswith("\r\n"):
                line += "\r\n"
            self.ser.write(line.encode())
        except serial.SerialException as e:
            print(f"[SERIAL] Ошибка записи: {e}. Закрываю порт.")
            try:
                self.ser.close()
            finally:
                self.ser = None

    def send_angles(self, x_deg, y_deg):
        """Протокол как у вас: X:...,Y:..."""
        self._send_command(f"X:{int(x_deg)},Y:{int(y_deg)}")

    def send_laser(self, on: bool):
        self.laser_on = on
        self._send_command("LAS 1" if on else "LAS 0")

    # ---------- Главный цикл ----------

    def run(self):
        print("Клавиши: Q-выход | L-лазер | C-центр | I/K-инверсия осей | S-сохранить инверсии | U/J +/-Kp")
        try:
            while True:
                frame = self._capture_and_process_frame()

                # детекция объектов
                results = self.model.predict(
                    frame, conf=self.settings.CONF_THRESHOLD, imgsz=self.settings.IMGSZ, verbose=False
                )
                target_pos = self._select_stable_target(results)  # стабильный выбор цели
                laser_pos = self._detect_laser_dot(frame)

                # состояние с лёгким гистерезисом
                state = self.state_machine.update(target_pos is not None, laser_pos is not None)
                now = time.time()
                if laser_pos:
                    self.last_laser_seen = now
                # если объект есть, а лазер «мелькнул» недавно — остаёмся в TRACK, чтобы не дёргало
                if state == "CHASE" and (now - self.last_laser_seen) < 0.3 and target_pos:
                    state = "TRACK"

                self._handle_state(state, target_pos, laser_pos)

                # overlay
                self._draw_overlay(frame, target_pos, laser_pos)
                cv2.imshow("Auto Aim", frame)

                if self._handle_keyboard():
                    break

        finally:
            self.cleanup()

    # ---------- Обработка кадра ----------

    def _capture_and_process_frame(self):
        frame = self.picam2.capture_array()
        if self.settings.ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.settings.ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.settings.ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _select_stable_target(self, results):
        """Стабильный выбор цели: ближняя к прошлой, иначе крупнейшая."""
        if not results or len(results[0].boxes) == 0:
            self.prev_target = None
            return None

        boxes = results[0].boxes
        cand = []
        for box in boxes:
            if int(box.cls.item()) in self.wanted_ids:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                cand.append((cx, cy, area))

        if not cand:
            self.prev_target = None
            return None

        if self.prev_target is None:
            best = max(cand, key=lambda t: t[2])  # старт: крупнейшая
        else:
            px, py = self.prev_target
            # ближе к прошлой + лёгкое предпочтение крупным
            best = min(cand, key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2 + 0.000001 * (1e8 / max(t[2], 1)))

        self.prev_target = (int(best[0]), int(best[1]))
        self.prev_target_time = time.time()
        return self.prev_target

    def _detect_laser_dot(self, frame):
        """Обнаружение яркой красной точки, отсекаем шум/блики."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # красный с высокими S и V — «яркая» точка
        lower1 = np.array([0, 150, 220]);  upper1 = np.array([10, 255, 255])
        lower2 = np.array([170,150, 220]); upper2 = np.array([179,255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 3 or a > 150:     # слишком маленькая/большая «клякса» — мимо
            return None
        M = cv2.moments(c)
        if not M["m00"]:
            return None
        x = int(M["m10"] / M["m00"]); y = int(M["m01"] / M["m00"])
        # игнорируем край кадра (часто блики)
        if x < 10 or x > self.w - 10 or y < 10 or y > self.h - 10:
            return None
        return (x, y)

    # ---------- Поведение по состояниям ----------

    def _handle_state(self, state, target_pos, laser_pos):
        if state == "IDLE":
            if self.laser_on:
                self.send_laser(False)
            # мягко возвращаемся к центру
            self.servo_x += (90 - self.servo_x) * 0.1
            self.servo_y += (90 - self.servo_y) * 0.1
            self._maybe_send_angles()
            self.search_radius = 0.0
            return

        if not self.laser_on:
            self.send_laser(True)

        if state == "TRACK":
            # ведём по ошибке между целью и лазером
            if target_pos and laser_pos:
                dx_px = target_pos[0] - laser_pos[0]
                dy_px = target_pos[1] - laser_pos[1]
                self._update_servos_from_pixel_error(dx_px, dy_px)
        elif state == "CHASE":
            # подводим лазер к цели по центру кадра
            if target_pos:
                dx_px = target_pos[0] - self.center_x
                dy_px = target_pos[1] - self.center_y
                self._update_servos_from_pixel_error(dx_px, dy_px)
        elif state == "SEARCH":
            # спиральный скан
            self.search_radius += self.settings.SEARCH_RADIUS_STEP / 10
            self.search_angle = (self.search_angle + self.settings.SEARCH_ANGLE_STEP) % 360
            ox = self.search_radius * np.cos(np.deg2rad(self.search_angle))
            oy = self.search_radius * np.sin(np.deg2rad(self.search_angle))
            tx = 90 + (ox if not self.invert_x else -ox)
            ty = 90 + (oy if not self.invert_y else -oy)
            self.servo_x = clamp(tx, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
            self.servo_y = clamp(ty, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
            self._maybe_send_angles()
            if self.search_radius > 25:
                self.search_radius = 0

    def _update_servos_from_pixel_error(self, dx_px, dy_px):
        """Перевод ошибки в пикселях → градусы → шаги серв с PI-контролем."""
        # мёртвая зона
        if abs(dx_px) < self.settings.DEADBAND_PX:
            dx_px = 0
        if abs(dy_px) < self.settings.DEADBAND_PX:
            dy_px = 0

        # пиксели → градусы через FOV
        error_deg_x = (dx_px / self.w) * self.settings.FOV_H_DEG
        error_deg_y = -(dy_px / self.h) * self.settings.FOV_V_DEG  # вверх +

        # инверсии осей (можно переключать в окне клавишами i/k)
        if self.invert_x:
            error_deg_x = -error_deg_x
        if self.invert_y:
            error_deg_y = -error_deg_y

        # PI-контроллер: шаги
        step_x = self.pid_x.calculate_step(error_deg_x, self.kp)
        step_y = self.pid_y.calculate_step(error_deg_y, self.kp)

        # ограничим абсолютный шаг
        step_x = clamp(step_x, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)
        step_y = clamp(step_y, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)

        # новые углы
        self.servo_x = clamp(self.servo_x + step_x, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)

        # отправка не чаще CMD_HZ
        self._maybe_send_angles()

    def _maybe_send_angles(self):
        now = time.time()
        if now >= self._next_tx:
            self.send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0 / self.settings.CMD_HZ

    # ---------- Визуализация / ввод ----------

    def _draw_overlay(self, frame, target_pos, laser_pos):
        if target_pos:
            cv2.circle(frame, target_pos, 8, (255, 0, 0), 2)     # цель (синяя)
        if laser_pos:
            cv2.circle(frame, laser_pos, 8, (0, 0, 255), 2)      # лазер (красная)
        cv2.circle(frame, (self.center_x, self.center_y), 4, (0, 255, 0), -1)  # центр (зелёный)

        info = f"S:{self.state_machine.state} | Kp:{self.kp:.2f} | INV:{int(self.invert_x)},{int(self.invert_y)}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _handle_keyboard(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('l'):
            self.send_laser(!self.laser_on)  # переключение
        elif key == ord('c'):
            self.servo_x, self.servo_y = 90.0, 90.0
            self.pid_x.reset(); self.pid_y.reset()
            self._maybe_send_angles()
        elif key == ord('i'):
            self.invert_x = not self.invert_x
            print(f"Invert X: {self.invert_x}")
        elif key == ord('k'):
            self.invert_y = not self.invert_y
            print(f"Invert Y: {self.invert_y}")
        elif key == ord('s'):
            self._save_calib()
        elif key == ord('u'):  # +Kp
            self.kp += 0.03
            print(f"Kp = {self.kp:.2f}")
        elif key == ord('j'):  # -Kp
            self.kp = max(0.0, self.kp - 0.03)
            print(f"Kp = {self.kp:.2f}")
        return False

    # ---------- Завершение ----------

    def cleanup(self):
        print("\nЗавершение...")
        try:
            if self.laser_on:
                self.send_laser(False)
        except Exception:
            pass

        if self.ser and self.ser.is_open:
            try:
                # мягко вернуться к центру
                for _ in range(10):
                    self.servo_x += (90 - self.servo_x) * 0.2
                    self.servo_y += (90 - self.servo_y) * 0.2
                    self._maybe_send_angles()
                    time.sleep(0.02)
                self.ser.close()
                print("[SERIAL] закрыт")
            except Exception as e:
                print(f"[SERIAL] Ошибка при закрытии: {e}")

        if self.picam2:
            self.picam2.stop()
            print("[CAMERA] остановлена")

        cv2.destroyAllWindows()
        self._save_calib()
        print("Готово.")


if __name__ == '__main__':
    controller = TurretController()
    controller.run()
