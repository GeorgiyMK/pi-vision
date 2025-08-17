#!/usr/bin/env python3
# Raspberry Pi 5 + Cam v3 -> Arduino (2 серво + лазер)
# Простое и надёжное ведение: если есть bbox — ВСЕГДА двигаем к его центру.
# Скан только при полном отсутствии детекции N кадров. Диагностика на экране.

import time, json, os
import cv2
import numpy as np
import serial, serial.tools.list_ports
from picamera2 import Picamera2
from ultralytics import YOLO

# ================= НАСТРОЙКИ =================
class Cfg:
    MODEL_PATH     = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]      # [] = любые
    CONF_THRESHOLD = 0.35
    IMGSZ          = 320
    FRAME_SIZE     = (640, 480)
    ROTATE_DEG     = 180

    # Камера — один раз выставили и не трогаем (стабильность)
    AE_ENABLE      = True
    ANALOG_GAIN    = 2.0          # 1.6..2.8 по освещению
    EXPOSURE_US    = None         # None при AE_ENABLE=True

    # Геометрия камеры (Cam v3 ~)
    FOV_H_DEG      = 66.0
    FOV_V_DEG      = 41.0

    # Серво
    MIN_ANGLE      = 30
    MAX_ANGLE      = 150

    # Управление (PD) и ограничения
    KP_X           = 0.25     # ↑ если вяло; ↓ если рывки
    KP_Y           = 0.25
    KD_X           = 0.15     # демпфер
    KD_Y           = 0.15
    DEAD_PX        = 8        # мёртвая зона (пиксели)
    MAX_STEP_DEG   = 4.0      # макс. шаг за цикл
    CMD_HZ         = 25.0     # частота отправки команд

    # «Липкость» рамки: при исчезновении детекции держим последнюю ещё N кадров
    BOX_KEEP_FR    = 6
    # Скан только если real_missing подряд
    MISS_FOR_SCAN  = 10

    # Лазер — просто он/офф по ошибке наведения
    LOCK_ERR_PX    = 25
    LOCK_ON_FR     = 4
    LOCK_OFF_FR    = 6

    # Скан по X (панорама)
    SCAN_MIN_ANGLE = 60
    SCAN_MAX_ANGLE = 120
    SCAN_SPEED_DEG = 1.2

    # Serial
    BAUDRATE       = 115200
    PROTOCOL_MODE  = "XY"     # "XY" → "X:..,Y:.."
    DEBUG_TX       = False

    CALIB_PATH     = "calib.json"
# ============================================

def clamp(v, a, b): return max(a, min(b, v))

# ———— выбор цели (ближайшая к прошлой; иначе крупнейшая) ————
def select_target(results, wanted_ids, prev_cxcy):
    if not results or len(results[0].boxes) == 0: return None
    best = None; best_cost = 1e18
    for box in results[0].boxes:
        cls = int(box.cls.item())
        if wanted_ids and cls not in wanted_ids: continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cx,cy = (x1+x2)/2, (y1+y2)/2
        area   = (x2-x1)*(y2-y1)
        if prev_cxcy is None:
            cost = -area              # берём крупнейшую
        else:
            px,py = prev_cxcy
            cost = (cx-px)**2 + (cy-py)**2 - 1e-4*area
        if cost < best_cost:
            best_cost = cost
            best = (int(cx), int(cy), int(x1), int(y1), int(x2), int(y2))
    return best

class Controller:
    def __init__(self):
        self.w, self.h = Cfg.FRAME_SIZE
        self.cx, self.cy = self.w//2, self.h//2

        self.picam2 = self._init_cam()
        self.model  = self._init_model()

        self.ser = None; self._connect_serial()
        self.servo_x = 90.0; self.servo_y = 90.0
        self._next_tx = 0.0

        # инверсии (можно менять во время работы клавишами i/k)
        self.invert_x = False; self.invert_y = False
        self._load_calib()

        # PD память
        self.prev_err_x_deg = 0.0; self.prev_err_y_deg = 0.0
        self.prev_t = time.time()

        # «липкая» рамка и счётчики
        self.last_box = None
        self.keep_frames = 0
        self.missing_frames_real = 0  # подряд кадров БЕЗ новой детекции

        # лазер и скан
        self.laser_on = False
        self.lock_ok = 0; self.lock_bad = 0
        self.scan_dir = 1

        # диагностика
        self.last_dbg = 0
        self.last_dx = 0; self.last_dy = 0

    # ———— камера ————
    def _init_cam(self):
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format":"RGB888","size":Cfg.FRAME_SIZE}))
        cam.start()
        try:
            ctrl = {"AeEnable": bool(Cfg.AE_ENABLE)}
            if Cfg.ANALOG_GAIN is not None: ctrl["AnalogueGain"] = float(Cfg.ANALOG_GAIN)
            if (Cfg.EXPOSURE_US is not None) and (not Cfg.AE_ENABLE):
                ctrl["ExposureTime"] = int(Cfg.EXPOSURE_US)
            cam.set_controls(ctrl)
            print(f"[CAM] AE={Cfg.AE_ENABLE} gain={Cfg.ANALOG_GAIN} exp_us={Cfg.EXPOSURE_US}")
        except Exception as e:
            print("[CAM] set_controls warn:", e)
        return cam

    # ———— YOLO ————
    def _init_model(self):
        m = YOLO(Cfg.MODEL_PATH)
        name2id = {v:k for k,v in m.names.items()}
        self.wanted_ids = [name2id[n] for n in Cfg.WANTED_CLASSES if n in name2id] if Cfg.WANTED_CLASSES else []
        print(f"[MODEL] ok; classes={Cfg.WANTED_CLASSES or 'ALL'}")
        return m

    # ———— serial ————
    def _find_port(self):
        for p in serial.tools.list_ports.comports():
            if "/ttyUSB" in p.device or "/ttyACM" in p.device:
                return p.device
            if "USB" in p.device or "ACM" in p.device:
                return p.device
        return None

    def _connect_serial(self):
        if self.ser and self.ser.is_open: return True
        port = self._find_port()
        if not port:
            print("[SERIAL] Arduino не найдена"); self.ser=None; return False
        try:
            self.ser = serial.Serial(port, Cfg.BAUDRATE, timeout=0.1)
            time.sleep(1.8)
            print(f"[SERIAL] connected {port}")
            return True
        except Exception as e:
            print("[SERIAL] open error:", e); self.ser=None; return False

    def _raw_send(self, line):
        if not self.ser or not self.ser.is_open:
            if not self._connect_serial(): return
        if not line.endswith("\r\n"): line += "\r\n"
        if Cfg.DEBUG_TX: print("[TX]", line.strip())
        try: self.ser.write(line.encode())
        except Exception as e:
            print("[SERIAL] write error:", e)
            try: self.ser.close()
            finally: self.ser=None

    def _send_angles(self, ax, ay):
        if Cfg.PROTOCOL_MODE.upper()=="XY":
            self._raw_send(f"X:{int(ax)},Y:{int(ay)}")
        else:
            self._raw_send(f"ANG {ax-90:.1f} {ay-90:.1f}")

    def _send_laser(self, on):
        self.laser_on = on; self._raw_send("LAS 1" if on else "LAS 0")

    def _limit_rate_and_send(self):
        now = time.time()
        if now >= self._next_tx:
            self._send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0/Cfg.CMD_HZ

    # ———— управление ————
    def _pd_step(self, dx_px, dy_px):
        # мёртвая зона
        if abs(dx_px) < Cfg.DEAD_PX: dx_px = 0
        if abs(dy_px) < Cfg.DEAD_PX: dy_px = 0

        # пиксели -> градусы камеры
        err_x_deg =  (dx_px / self.w) * Cfg.FOV_H_DEG   # вправо +
        err_y_deg = -(dy_px / self.h) * Cfg.FOV_V_DEG   # вверх +

        if self.invert_x: err_x_deg = -err_x_deg
        if self.invert_y: err_y_deg = -err_y_deg

        now = time.time()
        dt = max(0.001, now - self.prev_t)
        d_ex = (err_x_deg - self.prev_err_x_deg) / dt
        d_ey = (err_y_deg - self.prev_err_y_deg) / dt
        self.prev_err_x_deg, self.prev_err_y_deg, self.prev_t = err_x_deg, err_y_deg, now

        step_x = Cfg.KP_X*err_x_deg + Cfg.KD_X*d_ex
        step_y = Cfg.KP_Y*err_y_deg + Cfg.KD_Y*d_ey

        step_x = clamp(step_x, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)
        step_y = clamp(step_y, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send()

        # для отладки
        self.last_dx = dx_px
        self.last_dy = dy_px

    def _scan_step(self):
        self.servo_x += self.scan_dir * Cfg.SCAN_SPEED_DEG
        if self.servo_x >= Cfg.SCAN_MAX_ANGLE:
            self.servo_x = Cfg.SCAN_MAX_ANGLE; self.scan_dir = -1
        elif self.servo_x <= Cfg.SCAN_MIN_ANGLE:
            self.servo_x = Cfg.SCAN_MIN_ANGLE; self.scan_dir = 1
        self._limit_rate_and_send()

    # ———— калибровка инверсий ————
    def _load_calib(self):
        if os.path.exists(Cfg.CALIB_PATH):
            try:
                with open(Cfg.CALIB_PATH, "r") as f:
                    d = json.load(f)
                    self.invert_x = bool(d.get("invert_x", False))
                    self.invert_y = bool(d.get("invert_y", False))
                    print("[CALIB] loaded", d)
            except: pass
    def _save_calib(self):
        with open(Cfg.CALIB_PATH, "w") as f:
            json.dump({"invert_x": self.invert_x, "invert_y": self.invert_y}, f, indent=2)
        print("[CALIB] saved")

    # ———— helpers ————
    def _capture(self):
        frame = self.picam2.capture_array()
        if Cfg.ROTATE_DEG == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif Cfg.ROTATE_DEG == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif Cfg.ROTATE_DEG == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    # ———— цикл ————
    def run(self):
        print("Q=выход | I/K — инверт X/Y | U/J — Kp +/- | F/D — Kd +/- | S — сохранить инверсию | L — лазер вручную")
        try:
            while True:
                frame = self._capture()
                # YOLO
                results = self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
                box = select_target(results, self.wanted_ids,
                                    (self.last_box[0], self.last_box[1]) if self.last_box else None)

                if box is not None:
                    # есть свежая детекция — работаем ТОЛЬКО по ней
                    cx,cy,x1,y1,x2,y2 = box
                    self.last_box = box
                    self.keep_frames = 0
                    self.missing_frames_real = 0

                    dx = cx - self.cx
                    dy = cy - self.cy
                    self._pd_step(dx, dy)

                    # Лазер по гистерезису ошибки
                    err = max(abs(dx), abs(dy))
                    if err <= Cfg.LOCK_ERR_PX:
                        self.lock_ok += 1; self.lock_bad = 0
                    else:
                        self.lock_ok = max(0, self.lock_ok - 1); self.lock_bad += 1
                    if (not self.laser_on) and self.lock_ok >= Cfg.LOCK_ON_FR:
                        self._send_laser(True)
                    if self.laser_on and self.lock_bad >= Cfg.LOCK_OFF_FR:
                        self._send_laser(False)

                else:
                    # свежей детекции нет
                    self.missing_frames_real += 1
                    if self.last_box is not None and self.keep_frames < Cfg.BOX_KEEP_FR:
                        # держим последнюю рамку ещё немного (без движения — чтобы не усугублять)
                        self.keep_frames += 1
                    else:
                        self.last_box = None
                        # при долгой потере — аккуратно сканируем
                        if self.missing_frames_real >= Cfg.MISS_FOR_SCAN:
                            self._scan_step()
                        else:
                            # мягко к центру
                            self.servo_x += (90 - self.servo_x)*0.05
                            self.servo_y += (90 - self.servo_y)*0.05
                            self._limit_rate_and_send()
                    # лазер гасим по гистерезису
                    self.lock_ok = 0
                    self.lock_bad += 1
                    if self.laser_on and self.lock_bad >= Cfg.LOCK_OFF_FR:
                        self._send_laser(False)

                # overlay
                cv2.drawMarker(frame, (self.cx,self.cy), (255,255,255), cv2.MARKER_CROSS, 20, 2)
                if self.last_box:
                    cx,cy,x1,y1,x2,y2 = self.last_box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,200,0), 2)
                    cv2.circle(frame, (cx,cy), 5, (255,0,0), 2)
                cv2.putText(frame,
                            f"dx:{self.last_dx:+4.0f} dy:{self.last_dy:+4.0f}  "
                            f"ax:{self.servo_x:5.1f} ay:{self.servo_y:5.1f}  "
                            f"invX:{int(self.invert_x)} invY:{int(self.invert_y)}  "
                            f"Laser:{'ON' if self.laser_on else 'off'}",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.imshow("Object Centering — simple & robust", frame)

                # клавиатура
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('i'): self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
                elif key == ord('k'): self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
                elif key == ord('u'): Cfg.KP_X += .03; Cfg.KP_Y += .03; print("Kp:", Cfg.KP_X, Cfg.KP_Y)
                elif key == ord('j'): Cfg.KP_X = max(0.0, Cfg.KP_X-.03); Cfg.KP_Y = max(0.0, Cfg.KP_Y-.03); print("Kp:", Cfg.KP_X, Cfg.KP_Y)
                elif key == ord('f'): Cfg.KD_X += .03; Cfg.KD_Y += .03; print("Kd:", Cfg.KD_X, Cfg.KD_Y)
                elif key == ord('d'): Cfg.KD_X = max(0.0, Cfg.KD_X-.03); Cfg.KD_Y = max(0.0, Cfg.KD_Y-.03); print("Kd:", Cfg.KD_X, Cfg.KD_Y)
                elif key == ord('s'): self._save_calib()
                elif key == ord('l'): self._send_laser(!self.laser_on)  # ручной тест лазера

        finally:
            self._cleanup()

    def _cleanup(self):
        print("\nЗавершение...")
        try: self._send_laser(False)
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

if __name__ == "__main__":
    Controller().run()
