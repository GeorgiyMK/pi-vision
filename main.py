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
    DEADBAND_PX = 10                 # мёртвая зона пикселей
    MAX_STEP_DEG = 2.5               # ограничение шага за итерацию

    # --- PI (P + I) ---
    KP = 0.18                        # 0.12–0.25 подбирайте на месте
    KI = 0.015
    KD = 0.0
    SMOOTH_FACTOR = 0.6
    BIAS_LIMIT = 15.0

    # --- Состояния / гистерезис ---
    MISS_FRAMES_TO_SEARCH = 45       # ~1.5s при 30 FPS

    # --- Спираль поиска (SEARCH) ---
    SEARCH_RADIUS_STEP = 2.5
    SEARCH_ANGLE_STEP = 18

    # --- Serial ---
    BAUDRATE = 115200
    RECONNECT_DELAY_S = 2.0
    CMD_HZ = 25.0                    # частота отправки углов

    # --- Протокол команд к Arduino: "XY" (X:..,Y:..), "ANG" (ANG yaw pitch), или "DUAL" ---
    PROTOCOL_MODE = "ANG"

    # --- Калибровка/отладка ---
    CALIB_PATH = "calib.json"
    DEBUG_TX = False


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

    def calculate_step(self, error, Kp):
        self.smoothed_error = (
            self.smooth_factor * self.smoothed_error
            + (1 - self.smooth_factor) * error
        )
        self.bias = clamp(self.bias + self.Ki * self.smoothed_error,
                          -self.bias_limit, self.bias_limit)
        return Kp * self.smoothed_error + self.bias

    def reset(self):
        self.bias = 0.0
        self.smoothed_error = 0.0


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

        self.servo_x, self.servo_y = 90.0, 90.0
        self.laser_on = False

        self.kp = self.settings.KP
        self.pid_x = PIDController(self.settings.KI, self.settings.KD,
                                   self.settings.SMOOTH_FACTOR, self.settings.BIAS_LIMIT)
        self.pid_y = PIDController(self.settings.KI, self.settings.KD,
                                   self.settings.SMOOTH_FACTOR, self.settings.BIAS_LIMIT)

        self.state_machine = StateMachine()
        self.search_radius = 0.0
        self.search_angle = 0.0

        # Память цели, лазера
        self.prev_target = None  # (cx,cy,x1,y1,x2,y2)
        self.last_laser_seen = 0.0

        # лимит частоты отправки углов
        self._next_tx = 0.0

        # калибровка
        self.calib = self._load_calib()
        self._apply_calib()

    # ---------- Инициализация ----------

    def _init_camera(self):
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": self.settings.FRAME_SIZE}
        )
        picam2.configure(config)
        picam2.start()
        print("[CAMERA] ok")
        return picam2

    def _init_model(self):
        model = YOLO(self.settings.MODEL_PATH)
        self.name2id = {v: k for k, v in model.names.items()}
        self.wanted_ids = [self.name2id[n]
                           for n in self.settings.WANTED_CLASSES
                           if n in self.name2id]
        print(f"[MODEL] Загружено. Классы: {self.settings.WANTED_CLASSES}")
        return model

    def connect_serial(self):
        if self.ser and self.ser.is_open:
            return True
        port = next((p.device for p in serial.tools.list_ports.comports()
                     if "/ttyUSB" in p.device or "/ttyACM" in p.device), None)
        if not port:
            print("[SERIAL] Arduino не найдена.")
            self.ser = None
            return False
        self.ser = serial.Serial(port, self.settings.BAUDRATE, timeout=0.1)
        time.sleep(1.8)
        print(f"[SERIAL] Подключено: {port}")
        try:
            self.ser.reset_input_buffer()
            self._raw_send("PING")
            time.sleep(0.2)
            data = self.ser.read(100).decode(errors="ignore").strip()
            if data:
                print("[SERIAL] RX:", data)
        except Exception:
            pass
        return True

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

    def _raw_send(self, line: str):
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                return
        if not line.endswith("\r\n"):
            line += "\r\n"
        if self.settings.DEBUG_TX:
            print("[TX]", line.strip())
        try:
            self.ser.write(line.encode())
        except serial.SerialException as e:
            print(f"[SERIAL] Ошибка записи: {e}. Закрываю порт.")
            try:
                self.ser.close()
            finally:
                self.ser = None

    def send_angles(self, x_deg, y_deg):
        mode = self.settings.PROTOCOL_MODE.upper()
        if mode in ("DUAL", "XY"):
            self._raw_send(f"X:{int(x_deg)},Y:{int(y_deg)}")
        if mode in ("DUAL", "ANG"):
            yaw = x_deg - 90.0
            pitch = y_deg - 90.0
            self._raw_send(f"ANG {yaw:.1f} {pitch:.1f}")

    def send_laser(self, on: bool):
        self.laser_on = on
        self._raw_send("LAS 1" if on else "LAS 0")

    # ---------- Геометрия: ошибка «точка → прямоугольник» ----------

    @staticmethod
    def point_to_rect_error(px, py, x1, y1, x2, y2):
        """
        Возвращает (dx,dy) — насколько нужно сдвинуть точку (px,py),
        чтобы попасть внутрь прямоугольника [x1..x2]×[y1..y2].
        Если точка уже внутри — (0,0).
        """
        # ближайшая точка прямоугольника к (px,py)
        tx = clamp(px, x1, x2)
        ty = clamp(py, y1, y2)
        dx = tx - px
        dy = ty - py
        return int(dx), int(dy)

    # ---------- Главный цикл ----------

    def run(self):
        print("Q-выход | L-лазер | C-центр | I/K-инверсия осей | S-сохранить | U/J +/-Kp | P сменить протокол")
        try:
            while True:
                frame = self._capture_and_process_frame()

                # детекция объектов
                results = self.model.predict(
                    frame, conf=self.settings.CONF_THRESHOLD, imgsz=self.settings.IMGSZ, verbose=False
                )
                target = self._select_stable_target(results)   # (cx,cy,x1,y1,x2,y2) или None
                laser_pos = self._detect_laser_dot(frame)

                # состояние + лёгкий гистерезис
                state = self.state_machine.update(target is not None, laser_pos is not None)
                now = time.time()
                if laser_pos:
                    self.last_laser_seen = now
                if state == "CHASE" and (now - self.last_laser_seen) < 0.3 and target:
                    state = "TRACK"

                self._handle_state(state, target, laser_pos)

                # overlay
                self._draw_overlay(frame, target, laser_pos)
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
        """Стабильный выбор цели: ближняя к прошлой, иначе крупнейшая. Возвращает (cx,cy,x1,y1,x2,y2)."""
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
                cand.append((cx, cy, x1, y1, x2, y2, area))

        if not cand:
            self.prev_target = None
            return None

        if self.prev_target is None:
            best = max(cand, key=lambda t: t[6])  # крупнейшая
        else:
            px, py = self.prev_target[0], self.prev_target[1]
            best = min(cand, key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2 + 0.000001 * (1e8 / max(t[6], 1)))

        self.prev_target = (int(best[0]), int(best[1]), *[int(v) for v in best[2:6]])
        return self.prev_target

    def _detect_laser_dot(self, frame):
        """Обнаружение яркой красной точки, отсекаем шум/блики."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 150, 220]);  upper1 = np.array([10, 255, 255])
        lower2 = np.array([170,150,220]);  upper2 = np.array([179,255,255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 3 or a > 150:
            return None
        M = cv2.moments(c)
        if not M["m00"]:
            return None
        x = int(M["m10"] / M["m00"]); y = int(M["m01"] / M["m00"])
        if x < 10 or x > self.w - 10 or y < 10 or y > self.h - 10:
            return None
        return (x, y)

    # ---------- Поведение по состояниям ----------

    def _handle_state(self, state, target, laser_pos):
        # target = (tcx,tcy,x1,y1,x2,y2) или None
        if state == "IDLE":
            if self.laser_on:
                self.send_laser(False)
            self.servo_x += (90 - self.servo_x) * 0.1
            self.servo_y += (90 - self.servo_y) * 0.1
            self._maybe_send_angles()
            self.search_radius = 0.0
            return

        if not self.laser_on:
            self.send_laser(True)

        if state in ("TRACK", "CHASE") and target:
            tcx, tcy, x1, y1, x2, y2 = target

            if laser_pos:
                # Основная новинка: держим точку ВНУТРИ прямоугольника
                dx_px, dy_px = self.point_to_rect_error(laser_pos[0], laser_pos[1], x1, y1, x2, y2)
                # Если уже внутри — ошибки нет, держим как есть
            else:
                # Лазер не видим: подводим к центру бокса (быстрее всего завести внутрь)
                dx_px = tcx - self.center_x
                dy_px = tcy - self.center_y

            self._update_servos_from_pixel_error(dx_px, dy_px)

        elif state == "SEARCH":
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
        """Пиксели → градусы → шаги серв с PI-контролем."""
        if abs(dx_px) < self.settings.DEADBAND_PX:
            dx_px = 0
        if abs(dy_px) < self.settings.DEADBAND_PX:
            dy_px = 0

        error_deg_x = (dx_px / self.w) * self.settings.FOV_H_DEG
        error_deg_y = -(dy_px / self.h) * self.settings.FOV_V_DEG  # вверх +

        if self.invert_x:
            error_deg_x = -error_deg_x
        if self.invert_y:
            error_deg_y = -error_deg_y

        step_x = self.pid_x.calculate_step(error_deg_x, self.kp)
        step_y = self.pid_y.calculate_step(error_deg_y, self.kp)

        step_x = clamp(step_x, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)
        step_y = clamp(step_y, -self.settings.MAX_STEP_DEG, self.settings.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, self.settings.MIN_ANGLE, self.settings.MAX_ANGLE)

        self._maybe_send_angles()

    def _maybe_send_angles(self):
        now = time.time()
        if now >= self._next_tx:
            self.send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0 / self.settings.CMD_HZ

    # ---------- Визуализация / ввод ----------

    def _draw_overlay(self, frame, target, laser_pos):
        if target:
            tcx, tcy, x1, y1, x2, y2 = target
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)     # бокс цели
            cv2.circle(frame, (tcx, tcy), 6, (255, 0, 0), 2)               # центр цели (просто как подсказка)
        if laser_pos:
            cv2.circle(frame, laser_pos, 8, (0, 0, 255), 2)                 # точка лазера
        cv2.circle(frame, (self.center_x, self.center_y), 4, (0, 255, 0), -1)

        info = f"S:{self.state_machine.state} | Kp:{self.kp:.2f} | INV:{int(self.invert_x)},{int(self.invert_y)} | PROTO:{self.settings.PROTOCOL_MODE}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _handle_keyboard(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        elif key == ord('l'):
            self.send_laser(not self.laser_on)
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
        elif key == ord('p'):  # смена протокола
            modes = ["DUAL", "XY", "ANG"]
            idx = modes.index(self.settings.PROTOCOL_MODE)
            self.settings.PROTOCOL_MODE = modes[(idx + 1) % len(modes)]
            print(f"[SERIAL] Протокол -> {self.settings.PROTOCOL_MODE}")
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
