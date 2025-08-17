#!/usr/bin/env python3
# Raspberry Pi 5 + Camera Module 3 -> Arduino (2 серво + лазер)
# Логика: CHASE — быстрый прицел к центру bbox с учётом смещения лазера (dx,dy,dz) и оценки дистанции D.
#         TRACK — удерживаем лазер ВНУТРИ bbox по обратной связи (мягкий шаг).
#         SEARCH — спиральный обзор. Динамически переключаем экспозицию: OBJ (светло) / LASER (короткая).

import cv2, time, json, os, numpy as np
import serial, serial.tools.list_ports
from picamera2 import Picamera2
from ultralytics import YOLO
from math import tan, atan, atan2, sqrt, radians, degrees

# ================== НАСТРОЙКИ ==================
class Settings:
    # Модель / кадр
    MODEL_PATH = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]       # [] = любые классы
    CONF_THRESHOLD = 0.35
    IMGSZ = 320
    FRAME_SIZE = (640, 480)
    ROTATE_DEG = 180

    # Поле зрения камеры (Camera Module 3 ≈)
    FOV_H_DEG = 66.0
    FOV_V_DEG = 41.0

    # --- НОВОЕ: геометрия смещения лазера (в метрах, оси камеры: X вправо, Y вверх, Z вперёд) ---
    LASER_DX_M = -0.20   # лазер на 20 см левее камеры (влево -> отрицательно)
    LASER_DY_M =  0.00   # выше (+) / ниже (-) камеры
    LASER_DZ_M =  0.00   # спереди (+) / сзади (-) камеры (обычно 0)
    TARGET_DIST_M = 1.50 # оценка дистанции до цели для быстрого прицела (клавиши [ и ])

    # Серво
    MIN_ANGLE = 30
    MAX_ANGLE = 150

    # Шаги и усиления
    # TRACK (тонкая доводка по точке лазера)
    KP_TRACK       = 0.18
    MAX_STEP_TRACK = 2.5
    DEADBAND_PX    = 10

    # CHASE (быстрый выход на цель по центру bbox + офсет)
    MAX_STEP_CHASE = 8.0

    # Состояния
    MISS_FRAMES_TO_SEARCH = 20
    SEARCH_RADIUS_STEP = 2.5
    SEARCH_ANGLE_STEP  = 18

    # Экспозиция
    OBJ_AE_ENABLE   = True
    OBJ_GAIN        = 1.8          # 1.0..3.0 — «светлая» сцена для YOLO

    LASER_AE_ENABLE = False
    LASER_EXPOS_US  = 1200         # короткая выдержка для лазера (800–3000)
    LASER_GAIN      = 1.2

    # Серийный порт
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
    table = np.array([(i/255.0)**inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

class StateMachine:
    def __init__(self):
        self.state = "IDLE"; self.miss_dot = 0
    def update(self, obj_ok, dot_ok):
        prev = self.state
        if obj_ok and dot_ok:
            self.state = "TRACK"; self.miss_dot = 0
        elif obj_ok and not dot_ok:
            self.miss_dot += 1
            self.state = "CHASE" if self.miss_dot < Settings.MISS_FRAMES_TO_SEARCH else "SEARCH"
        else:
            self.state = "IDLE"; self.miss_dot = 0
        if self.state != prev: print(f"[STATE] {prev} -> {self.state}")
        return self.state

class TurretController:
    def __init__(self):
        self.s = Settings()
        self.picam2 = self._init_camera()
        self.model  = self._init_model()
        self.ser = None; self.connect_serial()

        self.w, self.h = self.s.FRAME_SIZE
        self.cx, self.cy = self.w//2, self.h//2

        self.fx = (self.w/2) / tan(radians(self.s.FOV_H_DEG/2))  # фокальные (пикс)
        self.fy = (self.h/2) / tan(radians(self.s.FOV_V_DEG/2))

        self.servo_x = 90.0; self.servo_y = 90.0
        self.laser_on = False
        self._next_tx = 0.0

        self.state = StateMachine()
        self.search_radius = 0.0; self.search_angle = 0.0

        self.calib = self._load_calib(); self._apply_calib()
        self.current_exp_mode = None  # "OBJ"/"LASER"

    # ---------- Камера ----------
    def _init_camera(self):
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format":"RGB888","size":Settings.FRAME_SIZE}))
        cam.start(); print("[CAMERA] started")
        return cam

    def set_exposure_mode(self, mode):
        if self.current_exp_mode == mode: return
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
        self.name2id = {v:k for k,v in m.names.items()}
        self.wanted_ids = [self.name2id[n] for n in self.s.WANTED_CLASSES if n in self.name2id] \
                          if self.s.WANTED_CLASSES else []
        print(f"[MODEL] YOLO ok; classes={self.s.WANTED_CLASSES or 'ALL'}")
        return m

    # ---------- Калибровка инверсий ----------
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
        with open(self.s.CALIB_PATH,"w") as f: json.dump(self.calib,f,indent=2)
        print("[CALIB] saved", self.calib)

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
            print("[SERIAL] write error:", e);
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

    # ---------- Геометрия ----------
    def pixel_to_cam_angles(self, px, py):
        # пиксели -> (yaw_cam, pitch_cam) в радианах (оси: yaw вправо +, pitch вверх +)
        yaw = atan((px - self.cx) / self.fx)
        pitch = atan(-(py - self.cy) / self.fy)
        return yaw, pitch

    def cam_to_laser_angles(self, yaw_cam, pitch_cam, Dm):
        # предполагаем, что цель на расстоянии Dm вперёд от камеры
        X = Dm * tan(yaw_cam)
        Y = Dm * tan(pitch_cam)
        Z = Dm
        dx, dy, dz = self.s.LASER_DX_M, self.s.LASER_DY_M, self.s.LASER_DZ_M
        Xl, Yl, Zl = (X - dx), (Y - dy), (Z - dz)
        yaw_l   = atan2(Xl, Zl)
        pitch_l = atan2(Yl, sqrt(Xl*Xl + Zl*Zl))
        return yaw_l, pitch_l

    def go_towards_angles(self, target_ax, target_ay, max_step):
        # Плавно идём к заданным абсолютным углам серв (в градусах)
        dx = clamp(target_ax - self.servo_x, -max_step, max_step)
        dy = clamp(target_ay - self.servo_y, -max_step, max_step)
        self.servo_x = clamp(self.servo_x + dx, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + dy, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self._limit_rate_and_send()

    # ---------- Вижн ----------
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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv,(0,120,180),(12,255,255))
        mask2 = cv2.inRange(hsv,(170,120,180),(179,255,255))
        mask_hsv = cv2.bitwise_or(mask1, mask2)
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
        M=cv2.moments(c)
        if not M["m00"]: return None
        x=int(M["m10"]/M["m00"]); y=int(M["m01"]/M["m00"])
        if x<6 or x>self.w-6 or y<6 or y>self.h-6: return None
        return (x,y)

    # ---------- Управление ----------
    @staticmethod
    def point_to_rect_error(px, py, x1, y1, x2, y2, margin=6):
        x1m, y1m = x1+margin, y1+margin
        x2m, y2m = x2-margin, y2-margin
        if x1m > x2m: x1m, x2m = x1, x2
        if y1m > y2m: y1m, y2m = y1, y2
        tx = clamp(px, x1m, x2m); ty = clamp(py, y1m, y2m)
        return int(tx-px), int(ty-py)

    def move_by_pixel_error(self, dx_px, dy_px, kp, max_step):
        if abs(dx_px) < self.s.DEADBAND_PX: dx_px = 0
        if abs(dy_px) < self.s.DEADBAND_PX: dy_px = 0
        err_x_deg =  (dx_px / self.w) * self.s.FOV_H_DEG
        err_y_deg = -(dy_px / self.h) * self.s.FOV_V_DEG
        if self.invert_x: err_x_deg = -err_x_deg
        if self.invert_y: err_y_deg = -err_y_deg
        step_x = clamp(kp*err_x_deg, -max_step, max_step)
        step_y = clamp(kp*err_y_deg, -max_step, max_step)
        self.servo_x = clamp(self.servo_x + step_x, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, self.s.MIN_ANGLE, self.s.MAX_ANGLE)
        self._limit_rate_and_send()

    # ---------- Состояния ----------
    def _handle_state(self, state, target, laser_pos):
        if state == "IDLE" or not target:
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            self.servo_x += (90 - self.servo_x)*0.05
            self.servo_y += (90 - self.servo_y)*0.05
            self._limit_rate_and_send()
            self.search_radius = 0
            return

        tcx,tcy,x1,y1,x2,y2 = target

        if state == "CHASE":
            # Светлая сцена для лучшей детекции объекта, лазер выключен
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)

            # --- НОВОЕ: быстрый прицел с учётом офсета и оценки дальности ---
            yaw_cam, pitch_cam = self.pixel_to_cam_angles(tcx, tcy)
            yaw_l, pitch_l = self.cam_to_laser_angles(yaw_cam, pitch_cam, self.s.TARGET_DIST_M)
            yaw_deg   = degrees(yaw_l)
            pitch_deg = degrees(pitch_l)
            if self.invert_x: yaw_deg   = -yaw_deg
            if self.invert_y: pitch_deg = -pitch_deg
            target_ax = clamp(90 + yaw_deg,   self.s.MIN_ANGLE, self.s.MAX_ANGLE)
            target_ay = clamp(90 + pitch_deg, self.s.MIN_ANGLE, self.s.MAX_ANGLE)

            # быстро движемся к целевым углам (ограниченный большой шаг)
            self.go_towards_angles(target_ax, target_ay, self.s.MAX_STEP_CHASE)

        elif state == "TRACK" and laser_pos:
            # Короткая выдержка — уверенно видим точку; лазер включён
            self.set_exposure_mode("LASER")
            if not self.laser_on: self.send_laser(True)
            dx,dy = self.point_to_rect_error(laser_pos[0], laser_pos[1], x1,y1,x2,y2, margin=6)
            self.move_by_pixel_error(dx, dy, kp=self.s.KP_TRACK, max_step=self.s.MAX_STEP_TRACK)

        elif state == "SEARCH":
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
            # TRACK без видимого лазера — временно ведём центр bbox (как CHASE-lite)
            self.set_exposure_mode("OBJ")
            if self.laser_on: self.send_laser(False)
            dx = tcx - self.cx; dy = tcy - self.cy
            self.move_by_pixel_error(dx, dy, kp=self.s.KP_TRACK, max_step= self.s.MAX_STEP_CHASE/2)

    # ---------- Главный цикл ----------
    def run(self):
        print("Q — выход | I/K — инверт X/Y | U/J — Kp(track) +/- | [/] — D +/- 0.2м | S — сохранить инверсию")
        try:
            while True:
                frame = self._capture()
                res = self.model.predict(frame, conf=self.s.CONF_THRESHOLD, imgsz=self.s.IMGSZ, verbose=False)
                target = self._select_target(res)
                laser_pos = self._detect_laser(frame)

                st = self.state.update(target is not None, laser_pos is not None)
                self._handle_state(st, target, laser_pos)

                # overlay
                draw = frame.copy()
                if target:
                    tcx,tcy,x1,y1,x2,y2 = target
                    cv2.rectangle(draw,(x1,y1),(x2,y2),(255,200,0),2)
                    cv2.circle(draw,(tcx,tcy),5,(255,0,0),2)
                if laser_pos:
                    cv2.circle(draw, laser_pos, 8, (0,0,255), 2)
                cv2.drawMarker(draw,(self.cx,self.cy),(255,255,255),cv2.MARKER_CROSS,20,2)
                info=(f"S:{self.state.state} Dot:{'yes' if laser_pos else 'no'}  "
                      f"Kp:{self.s.KP_TRACK:.2f}  D:{self.s.TARGET_DIST_M:.2f}m  "
                      f"offs=({self.s.LASER_DX_M:+.2f},{self.s.LASER_DY_M:+.2f},{self.s.LASER_DZ_M:+.2f}) "
                      f"invX:{int(self.invert_x)} invY:{int(self.invert_y)}")
                cv2.putText(draw, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.imshow("Auto Aim (offset + hold-in-box)", draw)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('i'): self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
                elif key == ord('k'): self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
                elif key == ord('u'): self.s.KP_TRACK += 0.03; print(f"Kp_track={self.s.KP_TRACK:.2f}")
                elif key == ord('j'): self.s.KP_TRACK = max(0.0, self.s.KP_TRACK-0.03); print(f"Kp_track={self.s.KP_TRACK:.2f}")
                elif key == ord('['): self.s.TARGET_DIST_M = max(0.3, self.s.TARGET_DIST_M - 0.2); print(f"D={self.s.TARGET_DIST_M:.2f} m")
                elif key == ord(']'): self.s.TARGET_DIST_M += 0.2; print(f"D={self.s.TARGET_DIST_M:.2f} m")
                elif key == ord('s'): self._save_calib()
                elif key == ord('l'): self.send_laser(not self.laser_on)

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
                    self._limit_rate_and_send(); time.sleep(0.02)
                self.ser.close(); print("[SERIAL] закрыт")
            except Exception as e:
                print("[SERIAL] close error:", e)
        if self.picam2: self.picam2.stop(); print("[CAMERA] остановлена")
        cv2.destroyAllWindows()
        try: self._save_calib()
        except: pass

if __name__ == "__main__":
    TurretController().run()
