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


# ------------------ НАСТРОЙКИ ------------------
class Settings:
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]       # какие классы отслеживаем
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180               # 0/90/180/270

    # Поле зрения камеры (Camera Module 3 ориентировочно)
    FOV_H_DEG = 66.0
    FOV_V_DEG = 41.0

    # Серво
    MIN_ANGLE = 30
    MAX_ANGLE = 150
    DEADBAND_PX = 10               # мёртвая зона в пикселях
    MAX_STEP_DEG = 2.0             # макс. шаг за цикл (градусы)
    CMD_HZ = 25.0                  # как часто отправлять углы

    # Пропорциональный коэффициент (скорость реакции)
    KP = 0.18

    # Состояния
    MISS_FRAMES_TO_SEARCH = 15
    SEARCH_RADIUS_STEP = 2.5
    SEARCH_ANGLE_STEP = 18

    # Serial
    BAUDRATE = 115200
    RECONNECT_DELAY_S = 2.0

    # Файлы
    CALIB_PATH = "calib.json"


# ------------------ ВСПОМОГАТЕЛЬНОЕ ------------------
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


class StateMachine:
    """IDLE / CHASE / TRACK / SEARCH"""
    def __init__(self):
        self.state = "IDLE"
        self.miss_dot_counter = 0

    def update(self, have_obj, have_dot):
        prev = self.state
        if have_obj and have_dot:
            self.state = "TRACK"
            self.miss_dot_counter = 0
        elif have_obj and not have_dot:
            self.miss_dot_counter += 1
            self.state = "SEARCH" if self.miss_dot_counter >= Settings.MISS_FRAMES_TO_SEARCH else "CHASE"
        else:
            self.state = "IDLE"
            self.miss_dot_counter = 0

        if self.state != prev:
            print(f"[STATE] {prev} -> {self.state}")
        return self.state


# ------------------ ОСНОВНОЙ КЛАСС ------------------
class TurretController:
    def __init__(self):
        self.s = Settings()
        self.picam2 = self._init_camera()
        self.model = self._init_model()
        self.ser = None
        self.connect_serial()

        self.w, self.h = self.s.FRAME_SIZE
        self.cx0, self.cy0 = self.w // 2, self.h // 2

        self.servo_x = 90.0
        self.servo_y = 90.0
        self.laser_on = False
        self.kp = self.s.KP
        self.next_tx = 0.0

        self.state = StateMachine()
        self.search_radius = 0.0
        self.search_angle = 0.0

        # калибровка инверсии осей
        self.calib = self._load_calib()
        self._apply_calib()

    # ---------- init ----------
    def _init_camera(self):
        try:
            p = Picamera2()
            cfg = p.create_preview_configuration(
                main={"format": "RGB888", "size": Settings.FRAME_SIZE}
            )
            p.configure(cfg)
            p.start()
            print("[CAMERA] ok")
            return p
        except Exception as e:
            print("[CAMERA] error:", e)
            raise

    def _init_model(self):
        m = YOLO(self.s.MODEL_PATH)
        self.name2id = {v: k for k, v in m.names.items()}
        self.wanted_ids = [self.name2id[n] for n in self.s.WANTED_CLASSES if n in self.name2id]
        print(f"[MODEL] loaded, classes: {self.s.WANTED_CLASSES}")
        return m

    # ---------- calib ----------
    def _load_calib(self):
        if os.path.exists(self.s.CALIB_PATH):
            try:
                with open(self.s.CALIB_PATH, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"invert_x": False, "invert_y": False}

    def _apply_calib(self):
        self.invert_x = self.calib.get("invert_x", False)
        self.invert_y = self.calib.get("invert_y", False)

    def _save_calib(self):
        self.calib = {"invert_x": self.invert_x, "invert_y": self.invert_y}
        try:
            with open(self.s.CALIB_PATH, "w") as f:
                json.dump(self.calib, f, indent=2)
            print("[CALIB] saved:", self.calib)
        except Exception as e:
            print("[CALIB] save error:", e)

    # ---------- serial ----------
    def connect_serial(self):
        if self.ser and self.ser.is_open:
            return True
        try:
            port = next((p.device for p in serial.tools.list_ports.comports()
                         if "USB" in p.device or "ACM" in p.device), None)
            if not port:
                print("[SERIAL] Arduino не найдена")
                self.ser = None
                return False
            self.ser = serial.Serial(port, self.s.BAUDRATE, timeout=0.1)
            time.sleep(1.8)
            print(f"[SERIAL] connected: {port}")
            return True
        except Exception as e:
            print("[SERIAL] error:", e)
            self.ser = None
            return False

    def _send(self, line):
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                return
        try:
            if not line.endswith("\r\n"):
                line += "\r\n"
            self.ser.write(line.encode())
        except Exception as e:
            print("[SERIAL] write error:", e)
            try:
                self.ser.close()
            finally:
                self.ser = None

    def send_angles(self, x, y):
        self._send(f"X:{int(x)},Y:{int(y)}")

    def send_laser(self, on):
        self.laser_on = on
        self._send("LAS 1" if on else "LAS 0")

    # ---------- helpers ----------
    @staticmethod
    def point_to_rect_error(px, py, x1, y1, x2, y2, margin=4):
        """Вектор до ближайшей точки прямоугольника.
        Если точка внутри (с запасом margin) — (0,0)."""
        x1m, y1m = x1 + margin, y1 + margin
        x2m, y2m = x2 - margin, y2 - margin
        if x1m > x2m: x1m, x2m = x1, x2
        if y1m > y2m: y1m, y2m = y1, y2
        tx = clamp(px, x1m, x2m)
        ty = clamp(py, y1m, y2m)
        return int(tx - px), int(ty - py)

    # ---------- main loop ----------
    def run(self):
        print("Q – выход, L – лазер, C – центр, I/K – инверт X/Y, U/J – +/-Kp, S – сохранить инверсию")
        try:
            while True:
                frame = self._capture_frame()

                # детекция
                results = self.model.predict(frame, conf=self.s.CONF_THRESHOLD, imgsz=self.s.IMGSZ, verbose=False)
                target = self._select_target(results)  # (cx,cy,x1,y1,x2,y2) или None
                laser = self._detect_laser(frame)

                st = self.state.update(target is not None, laser is not None)
                self._act(st, target, laser)

                # overlay
                self._draw_overlay(frame, target, laser)
                cv2.imshow("Auto Aim", frame)

                if self._handle_keys():
                    break
        finally:
            self.cleanup()

    def _capture_frame(self):
        frame = self.picam2.capture_array()
        if self.s.ROTATE_DEG == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.s.ROTATE_DEG == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.s.ROTATE_DEG == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _select_target(self, results):
        if not results or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best = None; max_area = 0
        for b in boxes:
            if int(b.cls.item()) in self.wanted_ids:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                    best = (cx, cy, int(x1), int(y1), int(x2), int(y2))
        return best

    def _detect_laser(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 150, 220]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170,150,220]); upper2 = np.array([179,255,255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 3 or a > 150: return None
        M = cv2.moments(c)
        if not M["m00"]: return None
        x = int(M["m10"] / M["m00"]); y = int(M["m01"] / M["m00"])
        if x < 8 or x > self.w-8 or y < 8 or y > self.h-8: return None
        return (x, y)

    # ---------- управление ----------
    def _act(self, state, target, laser):
        # IDLE: выключить лазер и мягко вернуться к центру
        if state == "IDLE":
            if self.laser_on: self.send_laser(False)
            self._soft_center()
            return

        if not self.laser_on:
            self.send_laser(True)

        if target is None:
            return

        tcx, tcy, x1, y1, x2, y2 = target

        if state == "TRACK" and laser:
            # держим лазер ВНУТРИ bbox
            dx_px, dy_px = self.point_to_rect_error(laser[0], laser[1], x1, y1, x2, y2, margin=4)
            self._move_from_pixel_error(dx_px, dy_px)

        elif state == "CHASE":
            # лазера нет — тянем к центру bbox
            dx_px = tcx - self.cx0
            dy_px = tcy - self.cy0
            self._move_from_pixel_error(dx_px, dy_px)

        elif state == "SEARCH":
            # простая спираль
            self.search_radius += self.s.SEARCH_RADIUS_STEP / 10
            self.search_angle = (self.search_angle + self.s.SEARCH_ANGLE_STEP) % 360
            ox = self.search_radius * np.cos(np.deg2rad(self.search_angle))
            oy = self.search_radius * np.sin(np.deg2rad(self.search_angle))
            tx = 90 + (ox if not self.invert_x else -ox)
            ty = 90 + (oy if not self.invert_y else -oy)
            self.servo_x = clamp(tx, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
            self.servo_y = clamp(ty, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
            self._maybe_send_angles()
            if self.search_radius > 25: self.search_radius = 0

    def _move_from_pixel_error(self, dx_px, dy_px):
        """Пиксели -> градусы -> ограниченный шаг -> отправка."""
        # мёртвая зона
        if abs(dx_px) < self.s.DEADBAND_PX: dx_px = 0
        if abs(dy_px) < self.s.DEADBAND_PX: dy_px = 0

        # в градусы, знак по осям
        err_x_deg = (dx_px / self.w) * self.s.FOV_H_DEG
        err_y_deg = -(dy_px / self.h) * self.s.FOV_V_DEG  # вверх +

        if self.invert_x: err_x_deg = -err_x_deg
        if self.invert_y: err_y_deg = -err_y_deg

        # пропорциональный шаг + ограничение
        step_x = clamp(self.kp * err_x_deg, -self.s.MAX_STEP_DEG, self.s.MAX_STEP_DEG)
        step_y = clamp(self.kp * err_y_deg, -self.s.MAX_STEP_DEG, self.s.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, self.s.MIN_ANGLE, self.s.MAX_ANGLE)

        self._maybe_send_angles()

    def _soft_center(self):
        self.servo_x += (90 - self.servo_x) * 0.1
        self.servo_y += (90 - self.servo_y) * 0.1
        self._maybe_send_angles()

    def _maybe_send_angles(self):
        now = time.time()
        if now >= self.next_tx:
            self.send_angles(self.servo_x, self.servo_y)
            self.next_tx = now + 1.0 / self.s.CMD_HZ

    # ---------- overlay / клавиатура ----------
    def _draw_overlay(self, frame, target, laser):
        if target:
            cx, cy, x1, y1, x2, y2 = target
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 2)
        if laser:
            cv2.circle(frame, laser, 7, (0, 0, 255), 2)
        cv2.circle(frame, (self.cx0, self.cy0), 4, (0, 255, 0), -1)
        info = f"S:{self.state.state} | Kp:{self.kp:.2f} | INV:{int(self.invert_x)},{int(self.invert_y)}"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return True
        elif key == ord('l'): self.send_laser(not self.laser_on)
        elif key == ord('c'):
            self.servo_x, self.servo_y = 90.0, 90.0
            self._maybe_send_angles()
        elif key == ord('i'):
            self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
        elif key == ord('k'):
            self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
        elif key == ord('u'):
            self.kp += 0.03; print(f"Kp={self.kp:.2f}")
        elif key == ord('j'):
            self.kp = max(0.0, self.kp - 0.03); print(f"Kp={self.kp:.2f}")
        elif key == ord('s'):
            self._save_calib()
        return False

    # ---------- завершение ----------
    def cleanup(self):
        print("\n[EXIT] cleaning up...")
        try:
            if self.laser_on: self.send_laser(False)
        except: pass

        if self.ser and self.ser.is_open:
            try:
                for _ in range(8):
                    self._soft_center()
                    time.sleep(0.02)
                self.ser.close()
                print("[SERIAL] closed")
            except Exception as e:
                print("[SERIAL] close error:", e)

        if self.picam2:
            self.picam2.stop()
            print("[CAMERA] stopped")
        cv2.destroyAllWindows()
        self._save_calib()
        print("done.")


if __name__ == '__main__':
    TurretController().run()
