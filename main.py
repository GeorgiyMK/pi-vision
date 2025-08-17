#!/usr/bin/env python3
# Raspberry Pi 5 + Cam v3 -> Arduino (2 серво + лазер)
# УСТОЙЧИВОЕ ВЕДЕНИЕ ЦЕЛИ + АВТОИНВЕРСИЯ ОСЕЙ (anti-runaway)
# Логика: ведём центр объекта в центр кадра (без обратной связи по пятну).
# При первом появлении цели — автоопределение знаков осей по малым «тычкам» серво.
# Если регулятор уводит в упор и ошибка растёт — автоинверт соответствующей оси.

import time, json, os
import cv2
import numpy as np
import serial, serial.tools.list_ports
from math import tan, radians
from picamera2 import Picamera2
from ultralytics import YOLO

# ================= НАСТРОЙКИ =================
class Cfg:
    # Модель и кадр
    MODEL_PATH     = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]      # [] = любые
    CONF_THRESHOLD = 0.35
    IMGSZ          = 320
    FRAME_SIZE     = (640, 480)
    ROTATE_DEG     = 180

    # Камера (фиксированная яркость, без переключений)
    AE_ENABLE      = True
    ANALOG_GAIN    = 2.0          # сделай 1.6..2.8 под сцену
    EXPOSURE_US    = None         # None = авто (если AE_ENABLE=True)

    # Геометрия камеры (Cam v3 ~)
    FOV_H_DEG      = 66.0
    FOV_V_DEG      = 41.0

    # Серво (абсолютные углы на Arduino 0..180)
    MIN_ANGLE      = 30
    MAX_ANGLE      = 150

    # Контроллер и ограничения
    KP_X           = 0.22         # П-коэффициенты
    KP_Y           = 0.22
    KD_X           = 0.16         # D-демпфер
    KD_Y           = 0.16
    DEAD_PX        = 10           # мёртвая зона в пикселях
    MAX_STEP_DEG   = 3.0          # макс. шаг за цикл (град)
    CMD_HZ         = 25.0         # частота команд в порт (Гц)

    # Стабилизация цели (alpha-beta)
    ALPHA          = 0.35
    BETA           = 0.55
    MAX_SKIP_FR    = 8            # держим цель без новых детекций
    MAX_JUMP_PX    = 140          # гейтинг по скачкам центра
    BOX_KEEP_FR    = 8            # «липкая» рамка на N кадров

    # Автоинверсия
    AUTO_INV_DELTA_DEG = 6.0      # «тычок» серво при калибровке
    AUTO_INV_WAIT_S    = 0.15     # ожидание после «тычка»
    AUTO_INV_SAMPLES   = 4        # усреднение по кадрам

    # Анти-разнос
    RUNAWAY_FRAMES     = 10       # за N кадров ошибка растёт у упора — переворачиваем ось
    RUNAWAY_EPS        = 6.0      # на сколько пикселей должна вырасти ошибка

    # Serial
    BAUDRATE       = 115200
    PROTOCOL_MODE  = "XY"         # "XY" или "ANG"
    DEBUG_TX       = False

    # Файл калибровки инверсий
    CALIB_PATH     = "calib.json"
# ============================================

def clamp(v, a, b): return max(a, min(b, v))

# ----------- Выбор стабильной цели -----------
def select_target_stable(results, wanted_ids, prev_cxcy):
    """Возвращает (cx,cy,x1,y1,x2,y2) ближайшей к прошлой цели, иначе крупнейшей."""
    if not results or len(results[0].boxes) == 0:
        return None
    cand = []
    for box in results[0].boxes:
        cls = int(box.cls.item())
        if wanted_ids and cls not in wanted_ids:
            continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cx,cy = (x1+x2)/2, (y1+y2)/2
        area = (x2-x1)*(y2-y1)
        cand.append((cx,cy,int(x1),int(y1),int(x2),int(y2),area))
    if not cand:
        return None
    if prev_cxcy is None:
        best = max(cand, key=lambda t: t[6])
    else:
        px,py = prev_cxcy
        best = min(cand, key=lambda t: (t[0]-px)**2 + (t[1]-py)**2 - 1e-4*t[6])
    return (int(best[0]), int(best[1]), best[2], best[3], best[4], best[5])

# --------- Alpha-Beta фильтр ---------
class AlphaBeta:
    def __init__(self, alpha, beta):
        self.alpha = alpha; self.beta = beta
        self.x = None; self.y = None
        self.vx = 0.0; self.vy = 0.0
        self.last_t = None
        self.miss = 0
    def update(self, meas, gate_px=9999, hold=False):
        t = time.time()
        dt = 0.001 if self.last_t is None else max(0.001, t - self.last_t)
        self.last_t = t
        if self.x is None:
            if meas is None: return None
            self.x, self.y = meas; self.vx = self.vy = 0.0; self.miss = 0
            return (int(self.x), int(self.y))
        # прогноз
        self.x += self.vx * dt; self.y += self.vy * dt
        if meas is not None:
            mx,my = meas
            if abs(mx-self.x) + abs(my-self.y) > gate_px:
                self.miss += 1
            else:
                rx = mx - self.x; ry = my - self.y
                self.x += self.alpha * rx; self.y += self.alpha * ry
                self.vx += (self.beta * rx) / dt; self.vy += (self.beta * ry) / dt
                self.miss = 0
        else:
            self.miss += 1
        if not hold and self.miss > Cfg.MAX_SKIP_FR:
            self.x = self.y = None; self.vx = self.vy = 0.0
            return None
        return (int(self.x), int(self.y))

# -------------- Основной класс --------------
class Controller:
    def __init__(self):
        self.cfg = Cfg()
        self.picam2 = self._init_cam()
        self.model  = self._init_model()
        self.ser    = None; self._connect_serial()

        self.w, self.h = self.cfg.FRAME_SIZE
        self.cx, self.cy = self.w//2, self.h//2

        self.servo_x = 90.0; self.servo_y = 90.0
        self._next_tx = 0.0

        self.invert_x = False; self.invert_y = False
        self._load_calib()

        self.filter = AlphaBeta(self.cfg.ALPHA, self.cfg.BETA)
        self.prev_box = None
        self.box_hold = 0

        self.prev_err_x = 0.0; self.prev_err_y = 0.0
        self.prev_time  = time.time()

        # автоинверсия + анти-разнос
        self.auto_inv_done = False
        self.prev_abs_err_x = None
        self.prev_abs_err_y = None
        self.runaway_x = 0
        self.runaway_y = 0

    # ---------- камера ----------
    def _init_cam(self):
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(
            main={"format":"RGB888","size":Cfg.FRAME_SIZE}
        ))
        cam.start()
        try:
            ctrl = {"AeEnable": bool(Cfg.AE_ENABLE)}
            if Cfg.ANALOG_GAIN is not None: ctrl["AnalogueGain"] = float(Cfg.ANALOG_GAIN)
            if (Cfg.EXPOSURE_US is not None) and (not Cfg.AE_ENABLE):
                ctrl["ExposureTime"] = int(Cfg.EXPOSURE_US)
            cam.set_controls(ctrl)
            print(f"[CAM] AE={Cfg.AE_ENABLE} gain={Cfg.ANALOG_GAIN} exp_us={Cfg.EXPOSURE_US}")
        except Exception as e:
            print("[CAM] set_controls warning:", e)
        print("[CAM] ready")
        return cam

    # ---------- модель ----------
    def _init_model(self):
        m = YOLO(Cfg.MODEL_PATH)
        name2id = {v:k for k,v in m.names.items()}
        self.wanted_ids = [name2id[n] for n in Cfg.WANTED_CLASSES if n in name2id] if Cfg.WANTED_CLASSES else []
        print(f"[MODEL] ok; classes={Cfg.WANTED_CLASSES or 'ALL'}")
        return m

    # ---------- serial ----------
    def _find_port(self):
        for p in serial.tools.list_ports.comports():
            if "/ttyUSB" in p.device or "/ttyACM" in p.device:
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
    def _limit_rate_and_send(self):
        now = time.time()
        if now >= self._next_tx:
            self._send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0/Cfg.CMD_HZ

    # ---------- автоинверсия осей ----------
    def _avg_target_center(self, n_frames=4):
        """Собирает несколько кадров и возвращает усреднённый (cx,cy) цели. None если цели нет."""
        acc = []; tries = 0
        while len(acc) < n_frames and tries < n_frames*3:
            frame = self._capture()
            res = self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
            box = select_target_stable(res, self.wanted_ids, None)
            if box is not None:
                acc.append((box[0], box[1]))
            tries += 1
        if not acc: return None
        mx = int(sum(p[0] for p in acc)/len(acc)); my = int(sum(p[1] for p in acc)/len(acc))
        return (mx, my)

    def _auto_invert_axes(self):
        """Малые «тычки» по X и Y -> смотрим, куда сместился центр бокса; выставляем invert_x/y."""
        print("[AUTOINV] start")
        base = self._avg_target_center(Cfg.AUTO_INV_SAMPLES)
        if base is None:
            print("[AUTOINV] нет цели — пропуск"); return False
        base_x, base_y = base

        # X: подберём знак тычка так, чтобы не упереться в предел
        dx = Cfg.AUTO_INV_DELTA_DEG if self.servo_x + Cfg.AUTO_INV_DELTA_DEG <= Cfg.MAX_ANGLE else -Cfg.AUTO_INV_DELTA_DEG
        self.servo_x = clamp(self.servo_x + dx, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send(); time.sleep(Cfg.AUTO_INV_WAIT_S)
        after = self._avg_target_center(Cfg.AUTO_INV_SAMPLES)
        if after is None:
            print("[AUTOINV] X: цель пропала — пропуск"); return False
        dcx = after[0] - base_x
        # знак отклика: R = sign(dcx * dservo)
        R = 1 if (dcx*dx) > 0 else -1 if (dcx*dx) < 0 else 0
        self.invert_x = (R > 0)  # если «в ту же сторону», инвертируем
        print(f"[AUTOINV] X: dx_servo={dx:+.1f} -> dcx={dcx:+d} => invert_x={self.invert_x}")

        # вернёмся обратно
        self.servo_x = clamp(self.servo_x - dx, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send(); time.sleep(0.05)

        # Y
        dy = Cfg.AUTO_INV_DELTA_DEG if self.servo_y + Cfg.AUTO_INV_DELTA_DEG <= Cfg.MAX_ANGLE else -Cfg.AUTO_INV_DELTA_DEG
        self.servo_y = clamp(self.servo_y + dy, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send(); time.sleep(Cfg.AUTO_INV_WAIT_S)
        after = self._avg_target_center(Cfg.AUTO_INV_SAMPLES)
        if after is None:
            print("[AUTOINV] Y: цель пропала — пропуск"); return False
        dcy = after[1] - base_y
        R = 1 if (dcy*dy) > 0 else -1 if (dcy*dy) < 0 else 0
        self.invert_y = (R > 0)
        print(f"[AUTOINV] Y: dy_servo={dy:+.1f} -> dcy={dcy:+d} => invert_y={self.invert_y}")

        # вернёмся обратно
        self.servo_y = clamp(self.servo_y - dy, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send(); time.sleep(0.05)

        print("[AUTOINV] done")
        self._save_calib()  # сохраним на будущее
        return True

    # ---------- управление ----------
    def _pd_step(self, err_px_x, err_px_y):
        # мёртвая зона
        if abs(err_px_x) < Cfg.DEAD_PX: err_px_x = 0
        if abs(err_px_y) < Cfg.DEAD_PX: err_px_y = 0

        # пиксели -> «градусы камеры»
        err_deg_x =  (err_px_x / self.w) * Cfg.FOV_H_DEG   # вправо +
        err_deg_y = -(err_px_y / self.h) * Cfg.FOV_V_DEG   # вверх +

        if self.invert_x: err_deg_x = -err_deg_x
        if self.invert_y: err_deg_y = -err_deg_y

        # PD
        now = time.time()
        dt  = max(0.001, now - self.prev_time)
        d_ex = (err_deg_x - self.prev_err_x) / dt
        d_ey = (err_deg_y - self.prev_err_y) / dt
        self.prev_err_x, self.prev_err_y, self.prev_time = err_deg_x, err_deg_y, now

        step_x = Cfg.KP_X*err_deg_x + Cfg.KD_X*d_ex
        step_y = Cfg.KP_Y*err_deg_y + Cfg.KD_Y*d_ey

        step_x = clamp(step_x, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)
        step_y = clamp(step_y, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)

        self._limit_rate_and_send()

        # ---- анти-разнос: если у упора и ошибка растёт, переворачиваем оси ----
        abs_err_x = abs(err_px_x); abs_err_y = abs(err_px_y)
        # X
        if (self.servo_x <= Cfg.MIN_ANGLE+0.5 or self.servo_x >= Cfg.MAX_ANGLE-0.5):
            if self.prev_abs_err_x is not None and abs_err_x > self.prev_abs_err_x + Cfg.RUNAWAY_EPS:
                self.runaway_x += 1
            else:
                self.runaway_x = 0
            if self.runaway_x >= Cfg.RUNAWAY_FRAMES:
                self.invert_x = not self.invert_x
                print(f"[ANTI] runaway X -> invert_x={self.invert_x}")
                self.runaway_x = 0
        self.prev_abs_err_x = abs_err_x
        # Y
        if (self.servo_y <= Cfg.MIN_ANGLE+0.5 or self.servo_y >= Cfg.MAX_ANGLE-0.5):
            if self.prev_abs_err_y is not None and abs_err_y > self.prev_abs_err_y + Cfg.RUNAWAY_EPS:
                self.runaway_y += 1
            else:
                self.runaway_y = 0
            if self.runaway_y >= Cfg.RUNAWAY_FRAMES:
                self.invert_y = not self.invert_y
                print(f"[ANTI] runaway Y -> invert_y={self.invert_y}")
                self.runaway_y = 0
        self.prev_abs_err_y = abs_err_y

    # ---------- цикл ----------
    def run(self):
        print("Q=выход | I/K — инверт X/Y | U/J — Kp +/- | F/D — Kd +/- | S — сохранить инверсию")
        try:
            while True:
                frame = self._capture()
                # Детекция
                results = self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
                meas_box = select_target_stable(results, self.wanted_ids,
                                                (self.prev_box[0], self.prev_box[1]) if self.prev_box else None)

                # «липкая рамка»
                if meas_box is None and self.prev_box is not None and self.box_hold < Cfg.BOX_KEEP_FR:
                    meas_cxcy = None; self.box_hold += 1
                else:
                    self.box_hold = 0
                    meas_cxcy = (meas_box[0], meas_box[1]) if meas_box else None
                    if meas_box: self.prev_box = meas_box

                # автоинверсия — один раз при наличии цели
                if (not self.auto_inv_done) and self.prev_box is not None:
                    ok = self._auto_invert_axes()
                    self.auto_inv_done = ok  # если не получилось, попробует ещё раз при следующем заходе

                # фильтр
                est = self.filter.update(meas_cxcy, gate_px=Cfg.MAX_JUMP_PX, hold=True)
                if est is None:
                    self.prev_box = None  # цель потеряна
                    # мягко к центру
                    self.servo_x += (90 - self.servo_x)*0.05
                    self.servo_y += (90 - self.servo_y)*0.05
                    self._limit_rate_and_send()
                else:
                    ex, ey = est
                    self._pd_step(ex - self.cx, ey - self.cy)

                # overlay
                cv2.drawMarker(frame, (self.cx,self.cy), (255,255,255), cv2.MARKER_CROSS, 20, 2)
                if self.prev_box:
                    cx,cy,x1,y1,x2,y2 = self.prev_box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,200,0), 2)
                    cv2.circle(frame, (cx,cy), 5, (255,0,0), 2)
                if est is not None:
                    cv2.circle(frame, (int(est[0]), int(est[1])), 5, (0,255,0), 2)

                info = (f"Kp=({Cfg.KP_X:.2f},{Cfg.KP_Y:.2f}) Kd=({Cfg.KD_X:.2f},{Cfg.KD_Y:.2f}) "
                        f"invX:{int(self.invert_x)} invY:{int(self.invert_y)} "
                        f"autoInv:{int(self.auto_inv_done)}")
                cv2.putText(frame, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.imshow("Stable Object Centering (auto-invert)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('i'): self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
                elif key == ord('k'): self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
                elif key == ord('u'): Cfg.KP_X += .03; Cfg.KP_Y += .03; print("Kp:", Cfg.KP_X, Cfg.KP_Y)
                elif key == ord('j'): Cfg.KP_X = max(0.0, Cfg.KP_X-.03); Cfg.KP_Y = max(0.0, Cfg.KP_Y-.03); print("Kp:", Cfg.KP_X, Cfg.KP_Y)
                elif key == ord('f'): Cfg.KD_X += .03; Cfg.KD_Y += .03; print("Kd:", Cfg.KD_X, Cfg.KD_Y)
                elif key == ord('d'): Cfg.KD_X = max(0.0, Cfg.KD_X-.03); Cfg.KD_Y = max(0.0, Cfg.KD_Y-.03); print("Kd:", Cfg.KD_X, Cfg.KD_Y)
                elif key == ord('s'): self._save_calib()

        finally:
            self._cleanup()

    # ---------- калибровка инверсий ----------
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

    # ---------- помощники ----------
    def _capture(self):
        frame = self.picam2.capture_array()
        if Cfg.ROTATE_DEG == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif Cfg.ROTATE_DEG == 90: frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif Cfg.ROTATE_DEG == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    # ---------- завершение ----------
    def _cleanup(self):
        print("\nЗавершение...")
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
