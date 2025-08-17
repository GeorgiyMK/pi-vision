#!/usr/bin/env python3
# Raspberry Pi 5 + Camera Module 3 + Arduino (2 серво + лазер)
# Стабильное наведение по центру объекта (YOLOv8) с самокалибровкой осей/масштаба,
# автоадаптацией Kp, анти-разносом, гистерезисом лазера и надёжной камерой/serial.

import os, json, time, math, traceback
import cv2
import numpy as np
import serial, serial.tools.list_ports

# -------- Camera --------
from picamera2 import Picamera2
try:
    from libcamera import Transform
except Exception:
    Transform = None  # fallback без поворота через libcamera

# -------- Detector --------
from ultralytics import YOLO


# ================= CONFIG =================
class Cfg:
    # Модель и кадр
    MODEL_PATH     = "yolov8n.pt"
    WANTED_CLASSES = ["cup"]        # [] = любые
    CONF_THRESHOLD = 0.35
    IMGSZ          = 320
    FRAME_SIZE     = (640, 480)
    ROTATE_DEG     = 180            # 0/90/180/270

    # Камера (фиксируем и не дёргаем на лету)
    AE_ENABLE      = True
    ANALOG_GAIN    = 2.0            # 1.6..2.8 по освещению
    EXPOSURE_US    = None           # None при AE_ENABLE=True
    AWB_ENABLE     = True

    # Серво (реальные пределы механики)
    MIN_ANGLE      = 30
    MAX_ANGLE      = 150

    # Управление (PD) и ограничения
    KP_INIT        = 0.25
    KD_INIT        = 0.15
    DEAD_PX        = 8
    MAX_STEP_DEG   = 4.0
    CMD_HZ         = 25.0

    # Стабилизация детекции
    BOX_KEEP_FR    = 6              # держим прошлую рамку N кадров
    MAX_JUMP_PX    = 160            # гейтинг по скачку центра
    MAX_SKIP_FR    = 8              # если столько кадров нет измерения — считаем цель потерянной

    # Самокалибровка осей/масштаба (град/пикс)
    CALI_POKE_DEG  = 6.0            # «тычок» серво
    CALI_WAIT_S    = 0.15
    CALI_SAMPLES   = 4              # усреднение
    CALI_MIN_DPX   = 3.0            # минимальный сдвиг пикселей, чтобы считать валидным

    # Автоадаптация Kp
    KP_MIN         = 0.12
    KP_MAX         = 0.6
    KP_STEP        = 0.02
    BIG_ERR_PX     = 120            # если ошибка крупная долго — Kp↑
    BIG_ERR_TIME_S = 1.0
    OSC_ZERO_X     = 6              # сколько смен знака/сек → Kp↓
    OSC_WINDOW_S   = 2.0

    # Лазер (включается, когда реально навелись)
    LOCK_ERR_PX    = 25
    LOCK_ON_FR     = 4
    LOCK_OFF_FR    = 6

    # Поиск (скан) при долгой потере
    MISS_FOR_SCAN  = 10
    SCAN_MIN_ANGLE = 60
    SCAN_MAX_ANGLE = 120
    SCAN_SPEED_DEG = 1.2

    # Serial
    BAUDRATE       = 115200
    PROTOCOL_MODE  = "XY"          # "XY" → "X:..,Y:.."
    DEBUG_TX       = False

    # Файл с калибровкой/параметрами
    CALIB_PATH     = "calib.json"
# =========================================


def clamp(v, a, b): return a if v < a else b if v > b else v


# ----------------- Выбор цели -----------------
def select_target(results, wanted_ids, prev_cxcy):
    """Возвращает (cx,cy,x1,y1,x2,y2) ближайшей к прошлой цели, иначе крупнейшей."""
    if not results or len(results[0].boxes) == 0:
        return None
    cand = []
    for box in results[0].boxes:
        cls = int(box.cls.item())
        if wanted_ids and cls not in wanted_ids: continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cx,cy = (x1+x2)/2, (y1+y2)/2
        area   = (x2-x1)*(y2-y1)
        cand.append((cx,cy,int(x1),int(y1),int(x2),int(y2),area))
    if not cand:
        return None
    if prev_cxcy is None:
        best = max(cand, key=lambda t: t[6])
    else:
        px,py = prev_cxcy
        best = min(cand, key=lambda t: (t[0]-px)**2 + (t[1]-py)**2 - 1e-4*t[6])
    return (int(best[0]), int(best[1]), best[2], best[3], best[4], best[5])


# ----------------- Класс контроллера -----------------
class Controller:
    def __init__(self):
        self.w, self.h = Cfg.FRAME_SIZE
        self.cx, self.cy = self.w//2, self.h//2

        # Калибровка/параметры по умолчанию
        self.invert_x = False
        self.invert_y = False
        self.deg_per_px_x = None    # вычислим в калибровке; если None — используем FOV оценку
        self.deg_per_px_y = None
        self.kp = Cfg.KP_INIT
        self.kd = Cfg.KD_INIT

        self._load_calib()

        # Камера и модель
        self.picam2 = self._init_cam()
        self.model  = self._init_model()

        # Serial
        self.ser = None; self._connect_serial()
        self.servo_x = 90.0; self.servo_y = 90.0
        self._next_tx = 0.0

        # Состояние ведения
        self.last_box = None
        self.keep_frames = 0
        self.missing_frames_real = 0

        # Память для PD
        self.prev_err_deg_x = 0.0
        self.prev_err_deg_y = 0.0
        self.prev_t = time.time()

        # Лазер с гистерезисом
        self.laser_on = False
        self.lock_ok = 0
        self.lock_bad = 0

        # Скан
        self.scan_dir = 1

        # Мониторинг для автоадаптации
        self.big_err_since = None
        self.sign_changes = []
        self.sign_last_x = 0
        self.sign_last_y = 0

        # Watchdog камеры (перезапуск при битых кадрах)
        self.bad_frames = 0

    # -------- Камера --------
    def _init_cam(self):
        cam = Picamera2()
        if Transform and Cfg.ROTATE_DEG in (0,90,180,270):
            cfg = cam.create_video_configuration(
                main={"format":"RGB888","size":Cfg.FRAME_SIZE},
                transform=Transform(rotation=Cfg.ROTATE_DEG)
            )
        else:
            cfg = cam.create_video_configuration(
                main={"format":"RGB888","size":Cfg.FRAME_SIZE}
            )
        cam.configure(cfg)
        cam.start()
        time.sleep(0.2)
        try:
            ctrl = {"AeEnable": bool(Cfg.AE_ENABLE), "AwbEnable": bool(Cfg.AWB_ENABLE)}
            if Cfg.ANALOG_GAIN is not None: ctrl["AnalogueGain"] = float(Cfg.ANALOG_GAIN)
            if (Cfg.EXPOSURE_US is not None) and (not Cfg.AE_ENABLE):
                ctrl["ExposureTime"] = int(Cfg.EXPOSURE_US)
            cam.set_controls(ctrl)
            print(f"[CAM] AE={Cfg.AE_ENABLE} gain={Cfg.ANALOG_GAIN} exp_us={Cfg.EXPOSURE_US} awb={Cfg.AWB_ENABLE}")
        except Exception as e:
            print("[CAM] set_controls warn:", e)
        print("[CAM] ready")
        return cam

    def _capture(self):
        frame = self.picam2.capture_array("main")
        # если пришёл «кусок» кадра — перезапустим камеру один раз
        if frame is None or frame.shape[0] < 100 or frame.shape[1] < 100:
            self.bad_frames += 1
            if self.bad_frames >= 2:
                print("[CAM] bad frame -> reinit camera")
                try:
                    self.picam2.stop()
                except: pass
                self.picam2 = self._init_cam()
                self.bad_frames = 0
                frame = self.picam2.capture_array("main")
        else:
            self.bad_frames = 0
        return frame

    # -------- YOLO --------
    def _init_model(self):
        m = YOLO(Cfg.MODEL_PATH)
        name2id = {v:k for k,v in m.names.items()}
        self.wanted_ids = [name2id[n] for n in Cfg.WANTED_CLASSES if n in name2id] if Cfg.WANTED_CLASSES else []
        print(f"[MODEL] ok; classes={Cfg.WANTED_CLASSES or 'ALL'}")
        return m

    # -------- Serial --------
    def _find_port(self):
        for p in serial.tools.list_ports.comports():
            dev = p.device
            if "/ttyUSB" in dev or "/ttyACM" in dev or "USB" in dev or "ACM" in dev:
                return dev
        return None

    def _connect_serial(self):
        if self.ser and self.ser.is_open: return True
        port = self._find_port()
        if not port:
            print("[SERIAL] Arduino не найдена")
            self.ser = None
            return False
        try:
            self.ser = serial.Serial(port, Cfg.BAUDRATE, timeout=0.1)
            time.sleep(1.8)
            print(f"[SERIAL] connected {port}")
            return True
        except Exception as e:
            print("[SERIAL] open error:", e)
            self.ser = None
            return False

    def _raw_send(self, line):
        if not self.ser or not self.ser.is_open:
            if not self._connect_serial(): return
        if not line.endswith("\r\n"): line += "\r\n"
        if Cfg.DEBUG_TX: print("[TX]", line.strip())
        try:
            self.ser.write(line.encode())
        except Exception as e:
            print("[SERIAL] write error:", e)
            try:
                self.ser.close()
            finally:
                self.ser = None

    def _send_angles(self, ax, ay):
        if Cfg.PROTOCOL_MODE.upper() == "XY":
            self._raw_send(f"X:{int(ax)},Y:{int(ay)}")
        else:
            self._raw_send(f"ANG {ax-90:.1f} {ay-90:.1f}")

    def _send_laser(self, on):
        self.laser_on = on
        self._raw_send("LAS 1" if on else "LAS 0")

    def _limit_rate_and_send(self):
        now = time.time()
        if now >= self._next_tx:
            self._send_angles(self.servo_x, self.servo_y)
            self._next_tx = now + 1.0 / Cfg.CMD_HZ

    # -------- Самокалибровка осей и масштаба --------
    def _avg_center(self, n_frames=4):
        acc = []; tries = 0
        while len(acc) < n_frames and tries < n_frames*3:
            frame = self._capture()
            res = self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
            box = select_target(res, self.wanted_ids, None)
            if box is not None:
                acc.append((box[0], box[1]))
            tries += 1
        if not acc: return None
        mx = sum(x for x,_ in acc)/len(acc); my = sum(y for _,y in acc)/len(acc)
        return (mx, my)

    def run_autocalib(self):
        """Определяем знак и град/пикс по X/Y малым шагом сервопривода.
           Требуется стабильная цель в кадре."""
        print("[CALIB] start...")
        base = self._avg_center(Cfg.CALI_SAMPLES)
        if base is None:
            print("[CALIB] нет цели — пропуск")
            return False
        bx, by = base

        # X
        d = Cfg.CALI_POKE_DEG if self.servo_x + Cfg.CALI_POKE_DEG <= Cfg.MAX_ANGLE else -Cfg.CALI_POKE_DEG
        self.servo_x = clamp(self.servo_x + d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(Cfg.CALI_WAIT_S)
        after = self._avg_center(Cfg.CALI_SAMPLES)
        if after is None:
            print("[CALIB] X: цель пропала — пропуск")
            return False
        dx_px = after[0] - bx
        if abs(dx_px) < Cfg.CALI_MIN_DPX:
            print("[CALIB] X: слишком мал сдвиг, оставляю прежний масштаб")
        # Направление: если +d дал +dx → экран двигается вправо при увеличении угла серво_x.
        self.invert_x = (dx_px * d > 0)  # инвертируем, чтобы «уменьшать» ошибку при положительном dx
        # Масштаб: град/пикс (сколько градусов сервоприводу нужно на 1 пикс ошибки)
        if abs(dx_px) >= Cfg.CALI_MIN_DPX:
            self.deg_per_px_x = abs(d / dx_px)
        print(f"[CALIB] X: d={d:+.1f}° -> dx={dx_px:+.1f}px  invert_x={self.invert_x}  deg/px_x={self.deg_per_px_x}")

        # Вернуть X
        self.servo_x = clamp(self.servo_x - d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(0.05)

        # Y
        d = Cfg.CALI_POKE_DEG if self.servo_y + Cfg.CALI_POKE_DEG <= Cfg.MAX_ANGLE else -Cfg.CALI_POKE_DEG
        self.servo_y = clamp(self.servo_y + d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(Cfg.CALI_WAIT_S)
        after = self._avg_center(Cfg.CALI_SAMPLES)
        if after is None:
            print("[CALIB] Y: цель пропала — пропуск")
            return False
        dy_px = after[1] - by
        if abs(dy_px) < Cfg.CALI_MIN_DPX:
            print("[CALIB] Y: слишком мал сдвиг, оставляю прежний масштаб")
        self.invert_y = (dy_px * d > 0)
        if abs(dy_px) >= Cfg.CALI_MIN_DPX:
            self.deg_per_px_y = abs(d / dy_px)
        print(f"[CALIB] Y: d={d:+.1f}° -> dy={dy_px:+.1f}px  invert_y={self.invert_y}  deg/px_y={self.deg_per_px_y}")

        # Вернуть Y
        self.servo_y = clamp(self.servo_y - d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(0.05)

        self._save_calib()
        print("[CALIB] done.")
        return True

    # -------- Управление (PD в градусах) --------
    def _px_to_deg(self, dx_px, dy_px):
        """Преобразование ошибок в пикселях в требуемые градусы камеры с учётом калибровки."""
        # Если калибровка масштаба не готова — оценка от FOV
        if self.deg_per_px_x is None:
            deg_per_px_x = (66.0 / self.w)      # оценка: FOV_H ~66°
        else:
            deg_per_px_x = self.deg_per_px_x
        if self.deg_per_px_y is None:
            deg_per_px_y = (41.0 / self.h)      # оценка: FOV_V ~41°
        else:
            deg_per_px_y = self.deg_per_px_y

        # учёт инверсий: хотим уменьшать ошибку
        err_deg_x = ( -dx_px if self.invert_x else dx_px ) * deg_per_px_x
        err_deg_y = ( -dy_px if self.invert_y else dy_px ) * deg_per_px_y
        return err_deg_x, err_deg_y

    def _pd_step(self, dx_px, dy_px):
        # мёртвая зона
        if abs(dx_px) < Cfg.DEAD_PX: dx_px = 0
        if abs(dy_px) < Cfg.DEAD_PX: dy_px = 0

        err_deg_x, err_deg_y = self._px_to_deg(dx_px, dy_px)

        now = time.time()
        dt = max(0.001, now - self.prev_t)
        d_ex = (err_deg_x - self.prev_err_deg_x) / dt
        d_ey = (err_deg_y - self.prev_err_deg_y) / dt
        self.prev_err_deg_x, self.prev_err_deg_y, self.prev_t = err_deg_x, err_deg_y, now

        step_x = clamp(self.kp*err_deg_x + self.kd*d_ex, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)
        step_y = clamp(self.kp*err_deg_y + self.kd*d_ey, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)

        self.servo_x = clamp(self.servo_x + step_x, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self.servo_y = clamp(self.servo_y + step_y, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        self._limit_rate_and_send()

        # Анти-разнос у упоров: если ошибка растёт у края — перевернуть ось
        self._anti_runaway(dx_px, dy_px)

        # Автоадаптация Kp (упрощённая и безопасная)
        self._auto_tune_kp(dx_px, dy_px, dt)

    def _anti_runaway(self, dx_px, dy_px):
        # Если у упора и |ошибка| растёт — инвертнём ось
        def upd(axis_angle, err_px, prev_err_deg, invert_flag_name):
            at_edge = (axis_angle <= Cfg.MIN_ANGLE+0.5) or (axis_angle >= Cfg.MAX_ANGLE-0.5)
            if not at_edge: return
            # если степень ошибки в пикселях увеличивается — вероятно, знак неверен
            if abs(err_px) > Cfg.DEAD_PX and abs(prev_err_deg) > 0.1:
                setattr(self, invert_flag_name, not getattr(self, invert_flag_name))
                print(f"[ANTI] invert {invert_flag_name} -> {getattr(self, invert_flag_name)}")
                self._save_calib()

        upd(self.servo_x, dx_px, self.prev_err_deg_x, "invert_x")
        upd(self.servo_y, dy_px, self.prev_err_deg_y, "invert_y")

    def _auto_tune_kp(self, dx_px, dy_px, dt):
        # 1) если ошибка большая долго — чуть ↑ Kp
        err_mag = max(abs(dx_px), abs(dy_px))
        now = time.time()
        if err_mag > Cfg.BIG_ERR_PX:
            if self.big_err_since is None: self.big_err_since = now
            elif (now - self.big_err_since) > Cfg.BIG_ERR_TIME_S and self.kp < Cfg.KP_MAX:
                self.kp = min(Cfg.KP_MAX, self.kp + Cfg.KP_STEP)
                self.big_err_since = now
                print(f"[KPAUTO] kp ↑ -> {self.kp:.2f}")
        else:
            self.big_err_since = None

        # 2) если часто меняем знак ошибки (качание) — чуть ↓ Kp
        def sign(v): return 1 if v>0 else (-1 if v<0 else 0)
        sx = sign(dx_px); sy = sign(dy_px)
        t_now = now
        if sx != 0 and sx != self.sign_last_x:
            self.sign_changes.append(t_now)
            self.sign_last_x = sx
        if sy != 0 and sy != self.sign_last_y:
            self.sign_changes.append(t_now)
            self.sign_last_y = sy
        # чистим старые события
        self.sign_changes = [t for t in self.sign_changes if (t_now - t) <= Cfg.OSC_WINDOW_S]
        # оценка «частоты» смен знака
        if len(self.sign_changes) >= Cfg.OSC_ZERO_X and self.kp > Cfg.KP_MIN:
            self.kp = max(Cfg.KP_MIN, self.kp - Cfg.KP_STEP)
            self.sign_changes.clear()
            print(f"[KPAUTO] kp ↓ -> {self.kp:.2f}")

    # -------- Скан --------
    def _scan_step(self):
        self.servo_x += self.scan_dir * Cfg.SCAN_SPEED_DEG
        if self.servo_x >= Cfg.SCAN_MAX_ANGLE:
            self.servo_x = Cfg.SCAN_MAX_ANGLE; self.scan_dir = -1
        elif self.servo_x <= Cfg.SCAN_MIN_ANGLE:
            self.servo_x = Cfg.SCAN_MIN_ANGLE; self.scan_dir = 1
        self._limit_rate_and_send()

    # -------- Файл калибровки --------
    def _load_calib(self):
        if os.path.exists(Cfg.CALIB_PATH):
            try:
                with open(Cfg.CALIB_PATH,"r") as f:
                    d = json.load(f)
                    self.invert_x = bool(d.get("invert_x", self.invert_x))
                    self.invert_y = bool(d.get("invert_y", self.invert_y))
                    self.deg_per_px_x = d.get("deg_per_px_x", self.deg_per_px_x)
                    self.deg_per_px_y = d.get("deg_per_px_y", self.deg_per_px_y)
                    self.kp = float(d.get("kp", self.kp))
                    self.kd = float(d.get("kd", self.kd))
                    print("[CALIB] loaded", d)
            except Exception as e:
                print("[CALIB] load warn:", e)

    def _save_calib(self):
        d = {
            "invert_x": self.invert_x,
            "invert_y": self.invert_y,
            "deg_per_px_x": self.deg_per_px_x,
            "deg_per_px_y": self.deg_per_px_y,
            "kp": self.kp,
            "kd": self.kd,
        }
        try:
            with open(Cfg.CALIB_PATH,"w") as f: json.dump(d,f,indent=2)
            print("[CALIB] saved", d)
        except Exception as e:
            print("[CALIB] save warn:", e)

    # -------- Главный цикл --------
    def run(self):
        print("Q=выход | I/K — инверт X/Y | R — авто-калибровка | U/J — Kp +/- | F/D — Kd +/- | L — лазер | S — сохранить")
        try:
            while True:
                frame = self._capture()
                results = self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
                box = select_target(results, self.wanted_ids,
                                    (self.last_box[0], self.last_box[1]) if self.last_box else None)

                if box is not None:
                    cx,cy,x1,y1,x2,y2 = box
                    # Гейтинг по «липкой» рамке/скачкам
                    if self.last_box is not None:
                        px,py = self.last_box[0], self.last_box[1]
                        if abs(cx-px)+abs(cy-py) > Cfg.MAX_JUMP_PX:
                            # слишком далеко — игнорируем этот кадр как выброс
                            box = None

                if box is not None:
                    self.last_box = box
                    self.keep_frames = 0
                    self.missing_frames_real = 0

                    dx = box[0] - self.cx
                    dy = box[1] - self.cy
                    self._pd_step(dx, dy)

                    # Лазер с гистерезисом
                    err = max(abs(dx), abs(dy))
                    if err <= Cfg.LOCK_ERR_PX:
                        self.lock_ok += 1; self.lock_bad = 0
                    else:
                        self.lock_ok = max(0, self.lock_ok-1); self.lock_bad += 1
                    if (not self.laser_on) and self.lock_ok >= Cfg.LOCK_ON_FR:
                        self._send_laser(True)
                    if self.laser_on and self.lock_bad >= Cfg.LOCK_OFF_FR:
                        self._send_laser(False)

                    # Автокалибровка при готовности масштаба (если ещё не делали)
                    if (self.deg_per_px_x is None or self.deg_per_px_y is None) and self.lock_ok >= Cfg.LOCK_ON_FR:
                        self.run_autocalib()

                else:
                    # нет fresh детекции
                    self.missing_frames_real += 1
                    if self.last_box is not None and self.keep_frames < Cfg.BOX_KEEP_FR:
                        self.keep_frames += 1  # держим ещё несколько кадров (без лишних движений)
                    else:
                        self.last_box = None
                        # при длительной потере — скан
                        if self.missing_frames_real >= Cfg.MISS_FOR_SCAN:
                            self._scan_step()
                        else:
                            # мягко к центру
                            self.servo_x += (90 - self.servo_x)*0.05
                            self.servo_y += (90 - self.servo_y)*0.05
                            self._limit_rate_and_send()
                    # лазер → off по гистерезису
                    self.lock_ok = 0; self.lock_bad += 1
                    if self.laser_on and self.lock_bad >= Cfg.LOCK_OFF_FR:
                        self._send_laser(False)

                # ---- Overlay ----
                cv2.drawMarker(frame, (self.cx,self.cy), (255,255,255), cv2.MARKER_CROSS, 20, 2)
                if self.last_box:
                    cx,cy,x1,y1,x2,y2 = self.last_box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,200,0), 2)
                    cv2.circle(frame, (cx,cy), 5, (255,0,0), 2)
                info = f"Kp:{self.kp:.2f} Kd:{self.kd:.2f} invX:{int(self.invert_x)} invY:{int(self.invert_y)} Laser:{'ON' if self.laser_on else 'off'}"
                cv2.putText(frame, info, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                cv2.imshow("Auto Aim (self-calibrating)", frame)

                # ---- Keys ----
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('i'): self.invert_x = not self.invert_x; print("invert_x:", self.invert_x)
                elif key == ord('k'): self.invert_y = not self.invert_y; print("invert_y:", self.invert_y)
                elif key == ord('r'): self.run_autocalib()
                elif key == ord('u'): self.kp = min(Cfg.KP_MAX, self.kp + Cfg.KP_STEP); print("kp:", self.kp)
                elif key == ord('j'): self.kp = max(Cfg.KP_MIN, self.kp - Cfg.KP_STEP); print("kp:", self.kp)
                elif key == ord('f'): self.kd += 0.03; print("kd:", self.kd)
                elif key == ord('d'): self.kd = max(0.0, self.kd - 0.03); print("kd:", self.kd)
                elif key == ord('l'): self._send_laser(not self.laser_on)
                elif key == ord('s'): self._save_calib()

        except Exception as e:
            print("[FATAL]", e); traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        print("\n[EXIT] shutting down...")
        try: self._send_laser(False)
        except: pass
        if self.ser and self.ser.is_open:
            try:
                for _ in range(8):
                    self.servo_x += (90 - self.servo_x)*0.25
                    self.servo_y += (90 - self.servo_y)*0.25
                    self._limit_rate_and_send(); time.sleep(0.02)
                self.ser.close(); print("[SERIAL] closed")
            except Exception as e:
                print("[SERIAL] close warn:", e)
        try: self.picam2.stop(); print("[CAM] stopped")
        except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Controller().run()
