#!/usr/bin/env python3
# Raspberry Pi 5 + Camera Module 3 -> Arduino (2 серво + лазер)
# Логика: в SEARCH/CHASE ведём центр bbox (AE=on, лазер off),
#         в TRACK удерживаем лазер ВНУТРИ bbox (AE=off, короткая выдержка, лазер on).

import cv2
import time
import json
import os
import numpy as np
import serial, serial.tools.list_ports
from picamera2 import Picamera2
from ultralytics import YOLO

# ================== НАСТРОЙКИ ==================
class Settings:
    # Модель / кадр
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]       # [] = любые классы
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180

    # Поле зрения камеры (примерно для Cam v3)
    FOV_H_DEG = 66.0
    FOV_V_DEG = 41.0

    # Серво
    MIN_ANGLE = 30
    MAX_ANGLE = 150
    MAX_STEP_DEG = 2.5
    DEADBAND_PX = 10

    # Контроллер
    KP = 0.18            # меняется клавишами U/J во время работы

    # Состояния
    MISS_FRAMES_TO_SEARCH = 20
    SEARCH_RADIUS_STEP = 2.5
    SEARCH_ANGLE_STEP  = 18

    # Экспозиция (два профиля)
    # Профиль для поиска объекта (светлая картинка)
    OBJ_AE_ENABLE   = True
    OBJ_GAIN        = 1.5          # 1.0..3.0 (подстрой под освещение)

    # Профиль для лазера (короткая выдержка, точка видна)
    LASER_AE_ENABLE = False
    LASER_EXPOS_US  = 1500         # 800..3000 мкс подстройкой
    LASER_GAIN      = 1.5

    # Gamma-подсветка предпросмотра (только для экрана)
    PREVIEW_GAMMA_OBJ   = 1.0      # SEARCH/CHASE
    PREVIEW_GAMMA_LASER = 0.8      # TRACK (слегка осветлим тёмный кадр)
    SHOW_GAMMA = True

    # Serial
    BAUDRATE = 115200
    CMD_HZ   = 25.0
    PROTOCOL_MODE = "XY"           # "XY" или "ANG"
    DEBUG_TX = False

    # Файлы
    CALIB_PATH = "calib.json"
# ===============================================

def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))

def apply_gamma(img, gamma):
    if abs(gamma-1.0) < 1e-3: return img
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ---------------- Состояния ----------------
class StateMachine:
    def __init__(self):
        self.state = "IDLE"
        self.miss_dot = 0

    def update(self, obj_ok, dot_ok):
        prev = self.state
        if obj_ok and dot_ok:
            self.state = "TRACK"; self.miss_dot = 0
        elif obj_ok and not dot_ok:
            self.miss_dot += 1
            self.state = "CHASE" if self.miss_dot < Settings.MISS_FRAMES_TO_SEARCH else "SEARCH"
        else:
            self.state = "IDLE"; self.miss_dot = 0
        if self.state != prev:
            print(f"[STATE] {prev} -> {self.state}")
        return self.state

# ------------- Контроллер -------------
class TurretController:
    def __init__(self):
        self.s = Settings()
        self.picam2 = self._init_camera()
        self.model  = self._init_model()

        self.ser = None
        self.connect_serial()

        self.w, self.h = self.s.FRAME_SIZE
        self.cx, self.cy = self.w//2, self.h//2

        self.servo_x = 90.0
        self.servo_y = 90.0
        self.kp = self.s.KP
        self.laser_on = False
        self._next_tx = 0.0

        self.state = StateMachine()

        self.search_radius = 0.0
        self.search_angle  = 0.0

        self.calib = self._load_calib()
        self._apply_calib()

        self.current_exp_mode = None  # "OBJ" или "LASER"

    # ---------- Камера ----------
    def _init_camera(self):
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format":"RGB888","size":Settings.FRAME_SIZE}))
        cam.start()
        print("[CAMERA] started")
        return cam

    def set_exposure_mode(self, mode):
        """mode: 'OBJ' (светло, AE on) или 'LASER' (короткая выдержка, AE off)."""
        if self.current_exp_mode == mode:
            return
        try:
            if mode == "OBJ":
                self.picam2.set_controls({
                    "AeEnable": True,
                    "AnalogueGain": float(self.s.OBJ_GAIN),
                })
            else:  # LASER
                self.picam2.set_controls({
                    "AeEnable": False,
                    "ExposureTime": int(self.s.LASER_EXPOS_US),
                    "AnalogueGain": float(self.s.LASER_GAIN),
                })
            self.current_exp_mode = mode
            print(f"[CAM] exposure -> {mode}")
        except Exception as e:
            print("[CAM] set_controls warning:", e)

    # ---------- Модель ----------
    def _init_model(self):
        m = YOLO(self.s.MODEL_PATH)
        self.name2id   = {v:k for k,v in m.names.items()}
        self.wanted_ids= [self.name2id[n] for n in self.s.WANTED_CLASSES if n in self.name2id] \
                         if self.s.WANTED_CLASSES else []
        print(f"[MODEL] YOLO ok; classes={self.s.WANTED_CLASSES or 'ALL'}")
        return m

    # ---------- Калибровка ----------
    def _load_calib(self):
        if os.path.exists(self.s.CALIB_PATH):
            try:
                with open(self.s.CALIB_PATH,"r") as f: return json.load(f)
            except: pass
        return {"invert_x":False,"invert_y":False}

    def _apply_calib(self):
        self.invert_x = self.calib.get("invert_x", False)
        self.invert_y = self.calib.get("invert_y", False)

    def _save_calib(self):
        self.calib = {"invert_x": self.invert_x, "invert_y": self.invert_y}
        try:
            with open(self.s.CALIB_PATH,"w") as f: json.dump(self.calib,f,indent=2)
            print("[CALIB] saved", self.calib)
        except Exception as e:
            print("[CALIB] save error:", e)

    # ---------- Serial ----------
    def connect_serial(self):
        if self.ser and self.ser.is_open: return True
        port = next((p.device for p in serial.tools.list_ports.comports()
                     if "/ttyUSB" in p.device or "/ttyACM" in p.device), None)
        if not port:
            print("[SERIAL] Arduino не найдена"); self.ser=None; return False
        try:
            self.ser = serial.Serial(port, self.s.BAUDRATE, timeout=0.1); time.sleep(1.8)
            print(f"[SERIAL] connected {port}"); return True
        except Exception as e:
            print("[SERIAL] open error:", e); self.ser=None; return False

    def _raw_send(self, line):
        if not self.ser or not self.ser.is_open:
            if not self.connect_serial(): return
        if not line.endswith("\r\n"): line += "\r\n"
        if self.s.DEBUG_TX: print("[TX]", line.strip())
        try: self.ser.write(line.encode())
        except Exception as e:
            print("[SERIAL] write error:", e)
            try: self.ser.close()
            finally: self.ser=None

    def send_angles(self, ax, ay):
        if self.s.PROTOCOL_MODE.upper()=="XY":
            self._raw_send(f"X:{int(ax)},Y:{int(ay)}")
        else:
            self._raw_send(f"ANG {ax-90:.1f} {ay-90:.1f}")

    def send_laser(self, on):
        self.laser_on = on; self._raw_send("LAS 1" if on else "LAS 0")

    def _limit_rate_and_send(self):
        now = time.time()
        if now >= self._next_tx:
            self.send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0 / self.s.CMD_HZ

    # ---------- Вспомогательные ----------
    @staticmethod
    def point_to_rect_error(px, py, x1, y1, x2, y2, margin=6):
        x1m, y1m = x1+margin, y1+margin
        x2m, y2m = x2-margin, y2-margin
        if x1m > x2m: x1m, x2m = x1, x2
        if y1m > y2m: y1m, y2m = y1, y2
        tx = clamp(px, x1m, x2m); ty = clamp(py, y1m, y2m)
        return int(tx-px), int(ty-py)   # 0,0 если уже внутри

    def _move_by_pixel_error(self, dx_px, dy_px):
        if abs(dx_px) < self.s.DEADBAND_PX: dx_px = 0
        if abs(dy_px) < self.s.DEADBAND_PX: dy_px = 0

        err_x_deg =  (dx_px / self.w) * self.s.FOV_H_DEG
        err_y_deg = -(dy_px / self.h) * self.s.FOV_V_DEG
        if self.invert_x: err_x_deg = -err_x_deg
        if self.invert_y: err_y_deg = -err_y_deg

        step_x = clamp(self.kp*err_x_deg, -self.s.MAX_STEP_DEG, self.s.MAX_STEP_DEG)
        step_y = clamp(self.kp*err_y_deg, -self.s.MAX_STEP_DEG, self.s.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self._limit_rate_and_send()

    # ---------- Детекция ----------
    def _capture(self):
        frame = self.picam2.capture_array()
        if self.s.ROTATE_DEG == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.s.ROTATE_DEG == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.s.ROTATE_DEG == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _select_target(self, results):
        if not results or len(results[0].boxes)==0: return None
        best=None; best_area=0
        for b in results[0].boxes:
            cls=int(b.cls.item())
            if self.wanted_ids and cls not in self.wanted_ids: continue
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            area=(x2-x1)*(y2-y1)
            if area>best_area:
                best_area=area; cx=int((x1+x2)/2); cy=int((y1+y2)/2)
                best=(cx,cy,int(x1),int(y1),int(x2),int(y2))
        return best

    def _detect_laser(self, frame):
        # HSV красный
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv,(0,120,180),(12,255,255))
        mask2 = cv2.inRange(hsv,(170,120,180),(179,255,255))
        mask_hsv = cv2.bitwise_or(mask1, mask2)

        # Яркий красный в RGB
        b,g,r = cv2.split(frame)
        rb = (r.astype(np.int16)-b.astype(np.int16)) > 40
        rg = (r.astype(np.int16)-g.astype(np.int16)) > 40
        bright = r > 200
        mask_rgb = (rb & rg & bright).astype(np.uint8)*255

        mask = cv2.bitwise_or(mask_hsv, mask_rgb)
        mask = cv2.medianBlur(mask,3)
        mask = cv2.dilate(mask, np.ones((3,3),np.uint8), 1)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea); a=cv2.contourArea(c)
        if a<3 or a>250: return None
        M=cv2.moments(c); if_not = (M["m00"]==0)
        if if_not: return None
        x=int(M["m10"]/M["m00"]); y=int(M["m01"]/M["m00"])
        if x<6 or x>self.w-6 or y<6 or y>self.h-6: return None
        return (x,y)

    # ---------- Обработка состояний ----------
    def _handle_state(self, state, target, laser_pos):
        if state in ("IDLE",) or not target:
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            # мягко к центру
            self.servo_x += (90 - self.servo_x)*0.05
            self.servo_y += (90 - self.servo_y)*0.05
            self._limit_rate_and_send()
            self.search_radius = 0
            return

        tcx,tcy,x1,y1,x2,y2 = target

        if state == "CHASE":
            # светлая сцена, лазер выключен (не мешает AE)
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            dx = tcx - self.cx; dy = tcy - self.cy
            self._move_by_pixel_error(dx, dy)

        elif state == "TRACK" and laser_pos:
            # удерживаем лазер внутри bbox
            self.set_exposure_mode("LASER")
            if not self.laser_on: self.send_laser(True)
            dx,dy = self.point_to_rect_error(laser_pos[0], laser_pos[1], x1,y1,x2,y2, margin=6)
            self._move_by_pixel_error(dx, dy)

        elif state == "SEARCH":
            # обзор по спирали; лазер лучше выключить
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            self.search_radius += self.s.SEARCH_RADIUS_STEP/10.0
            self.search_angle = (self.search_angle + self.s.SEARCH_ANGLE_STEP) % 360
            ox = self.search_radius*np.cos(np.deg2rad(self.search_angle))
            oy = self.search_radius*np.sin(np.deg2rad(self.search_angle))
            tx = 90 + (ox if not self.invert_x else -ox)
            ty = 90 + (oy if not self.invert_y else -oy)
            self.servo_x = clamp(tx, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
            self.servo_y = clamp(ty, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
            self._limit_rate_and_send()
            if self.search_radius > 25: self.search_radius = 0

        else:
            # TRACK без лазерной точки — временно ведём центр bbox
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            dx = tcx - self.cx; dy = tcy - self.cy
            self._move_by_pixel_error(dx, dy)

    # ---------- Главный цикл ----------
    def run(self):
        print("Q — выход | I/K — инверт X/Y | U/J — Kp +/- | S — сохранить инверсию | G — toggle gamma")
        show_gamma = self.s.SHOW_GAMMA
        try:
            while True:
                frame = self._capture()

                # детекция объекта
                res = self.model.predict(frame, conf=self.s.CONF_THRESHOLD, imgsz=self.s.IMGSZ, verbose=False)
                target = self._select_target(res)

                # детекция лазера (быстрее делать ПОСЛЕ выбора цели, но здесь ок)
                laser_pos = self._detect_laser(frame)

                st = self.state.update(target is not None, laser_pos is not None)
                self._handle_state(st, target, laser_pos)

                # overlay
                draw = frame.copy()
                if st == "TRACK":
                    if show_gamma:
                        draw = apply_gamma(draw, self.s.PREVIEW_GAMMA_LASER)
                else:
                    if show_gamma:
                        draw = apply_gamma(draw, self.s.PREVIEW_GAMMA_OBJ)

                if target:
                    tcx,tcy,x1,y1,x2,y2 = target
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(255,200,0),2)
                    cv2.circle(draw,(tcx,tcy),5,(255,0,0),2)
                if laser_pos:
                    cv2.circle(draw, laser_pos, 8, (0,0,255), 2)

                cv2.drawMarker(draw,(self.cx,self.cy),(255,255,255),cv2.MARKER_CROSS,20,2)
                exp = "OBJ(AE)" if self.current_exp_mode=="OBJ" else "LASER(man)"
                info=f"S:{self.state.state} Kp:{self.kp:.2f} invX:{int(self.invert_x)} invY:{int(self.invert_y)} Dot:{'yes' if laser_pos else 'no'} AE:{exp}"
                cv2.putText(draw, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("Auto Aim", draw)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('i'): self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
                elif key == ord('k'): self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
                elif key == ord('u'): self.kp += 0.03; print(f"Kp={self.kp:.2f}")
                elif key == ord('j'): self.kp = max(0.0, self.kp-0.03); print(f"Kp={self.kp:.2f}")
                elif key == ord('s'): self._save_calib()
                elif key == ord('g'): show_gamma = not show_gamma; print("gamma:", show_gamma)

        finally:
            self.cleanup()

    # ---------- Завершение ----------
    def cleanup(self):
        print("\nЗавершение...")
        try:
            if self.laser_on: self.send_laser(False)
        except: pass

        if self.ser and self.ser.is_open:
            try:
                for _ in range(10):
                    self.servo_x += (90 - self.servo_x)*0.2
                    self.servo_y += (90 - self.servo_y)*0.2
                    self._limit_rate_and_send()
                    time.sleep(0.02)
                self.ser.close()
                print("[SERIAL] закрыт")
            except Exception as e:
                print("[SERIAL] close error:", e)

        if self.picam2: self.picam2.stop(); print("[CAMERA] остановлена")
        cv2.destroyAllWindows()
        self._save_calib()
        print("Готово.")

if __name__ == "__main__":
    TurretController().run()
