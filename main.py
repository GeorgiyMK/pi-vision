#!/usr/bin/env python3
# Raspberry Pi 5 + Cam v3 + Arduino (2 серво + лазер)
# Жёстко-устойчивое ведение: фиксим X двигается, Y молчит; лазер не включается.
# Фишки: перестановка осей, дубль-команды для Y и лазера, самотест серв, автокалибровка, гистерезис.

import os, json, time, traceback
import cv2
import numpy as np
import serial, serial.tools.list_ports
from picamera2 import Picamera2
try:
    from libcamera import Transform
except Exception:
    Transform = None
from ultralytics import YOLO

class Cfg:
    # Модель/кадр
    MODEL_PATH="yolov8n.pt"; WANTED_CLASSES=["cup"]; CONF_THRESHOLD=0.35
    IMGSZ=320; FRAME_SIZE=(640,480); ROTATE_DEG=180

    # Камера стабильно (без переключений во время работы)
    AE_ENABLE=True; AWB_ENABLE=True; ANALOG_GAIN=2.0; EXPOSURE_US=None  # None при AE_ENABLE=True

    # Серво
    MIN_ANGLE=30; MAX_ANGLE=150; CMD_HZ=25.0

    # Контроллер
    KP_INIT=0.25; KD_INIT=0.15; DEAD_PX=8; MAX_STEP_DEG=4.0

    # Стабилизация детекции
    BOX_KEEP_FR=6; MAX_JUMP_PX=160; MAX_SKIP_FR=8

    # Самокалибровка (тычки)
    CALI_POKE_DEG=6.0; CALI_WAIT_S=0.15; CALI_SAMPLES=4; CALI_MIN_DPX=3.0

    # Лазер (гистерезис)
    LOCK_ERR_PX=25; LOCK_ON_FR=4; LOCK_OFF_FR=6

    # Поиск
    MISS_FOR_SCAN=10; SCAN_MIN_ANGLE=60; SCAN_MAX_ANGLE=120; SCAN_SPEED_DEG=1.2

    # Serial / протокол
    BAUDRATE=115200
    PROTOCOL_MODE="XY"   # "XY" -> "X:###,Y:###"
    SEND_Y_TWICE=True    # Дублировать Y отдельной строкой "Y:###"
    SEND_ALT_LAS=True    # Лазер шлём "LAS 1", "LAS:1", "LASER 1"
    DEBUG_TX=False

    CALIB_PATH="calib.json"

def clamp(v,a,b): return a if v<a else b if v>b else v

def select_target(results, wanted_ids, prev_cxcy):
    if not results or len(results[0].boxes)==0: return None
    best=None; best_cost=1e18
    for box in results[0].boxes:
        cls=int(box.cls.item())
        if wanted_ids and cls not in wanted_ids: continue
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        cx,cy=(x1+x2)/2,(y1+y2)/2; area=(x2-x1)*(y2-y1)
        if prev_cxcy is None: cost=-area
        else:
            px,py=prev_cxcy; cost=(cx-px)**2+(cy-py)**2-1e-4*area
        if cost<best_cost:
            best_cost=cost; best=(int(cx),int(cy),int(x1),int(y1),int(x2),int(y2))
    return best

class Controller:
    def __init__(self):
        self.w,self.h = Cfg.FRAME_SIZE; self.cx,self.cy = self.w//2,self.h//2

        # Калибровка/параметры
        self.invert_x=False; self.invert_y=False
        self.swap_axes=False            # НОВОЕ: перестановка осей X<->Y (клавиша X)
        self.deg_per_px_x=None; self.deg_per_px_y=None
        self.kp=Cfg.KP_INIT; self.kd=Cfg.KD_INIT
        self._load_calib()

        # Камера/модель
        self.picam2=self._init_cam()
        self.model=self._init_model()

        # Serial
        self.ser=None; self.port=None; self._connect_serial()
        self.servo_x=90.0; self.servo_y=90.0; self._next_tx=0.0

        # Состояние
        self.last_box=None; self.keep_frames=0; self.missing_frames_real=0
        self.prev_err_deg_x=0.0; self.prev_err_deg_y=0.0; self.prev_t=time.time()

        # Лазер
        self.laser_on=False; self.lock_ok=0; self.lock_bad=0

        # Скан
        self.scan_dir=1

        # Защита камеры
        self.bad_frames=0

    # ---------- Camera ----------
    def _init_cam(self):
        cam=Picamera2()
        if Transform and Cfg.ROTATE_DEG in (0,90,180,270):
            cfg=cam.create_video_configuration(main={"format":"RGB888","size":Cfg.FRAME_SIZE},
                                               transform=Transform(rotation=Cfg.ROTATE_DEG))
        else:
            cfg=cam.create_video_configuration(main={"format":"RGB888","size":Cfg.FRAME_SIZE})
        cam.configure(cfg); cam.start(); time.sleep(0.2)
        try:
            ctrl={"AeEnable":bool(Cfg.AE_ENABLE),"AwbEnable":bool(Cfg.AWB_ENABLE)}
            if Cfg.ANALOG_GAIN is not None: ctrl["AnalogueGain"]=float(Cfg.ANALOG_GAIN)
            if (Cfg.EXPOSURE_US is not None) and (not Cfg.AE_ENABLE): ctrl["ExposureTime"]=int(Cfg.EXPOSURE_US)
            cam.set_controls(ctrl)
            print(f"[CAM] AE={Cfg.AE_ENABLE} gain={Cfg.ANALOG_GAIN} exp_us={Cfg.EXPOSURE_US} awb={Cfg.AWB_ENABLE}")
        except Exception as e:
            print("[CAM] set_controls warn:", e)
        return cam

    def _capture(self):
        frame=self.picam2.capture_array("main")
        if frame is None or frame.shape[0]<100 or frame.shape[1]<100:
            self.bad_frames+=1
            if self.bad_frames>=2:
                print("[CAM] bad frame -> reinit")
                try: self.picam2.stop()
                except: pass
                self.picam2=self._init_cam(); self.bad_frames=0
                frame=self.picam2.capture_array("main")
        else:
            self.bad_frames=0
        return frame

    # ---------- YOLO ----------
    def _init_model(self):
        m=YOLO(Cfg.MODEL_PATH)
        name2id={v:k for k,v in m.names.items()}
        self.wanted_ids=[name2id[n] for n in Cfg.WANTED_CLASSES if n in name2id] if Cfg.WANTED_CLASSES else []
        print(f"[MODEL] ok; classes={Cfg.WANTED_CLASSES or 'ALL'}")
        return m

    # ---------- Serial ----------
    def _find_port(self):
        for p in serial.tools.list_ports.comports():
            dev=p.device
            if "/ttyUSB" in dev or "/ttyACM" in dev or "USB" in dev or "ACM" in dev:
                return dev
        return None
    def _connect_serial(self):
        if self.ser and self.ser.is_open: return True
        self.port=self._find_port()
        if not self.port:
            print("[SERIAL] Arduino не найдена"); self.ser=None; return False
        try:
            self.ser=serial.Serial(self.port, Cfg.BAUDRATE, timeout=0.1); time.sleep(1.8)
            print(f"[SERIAL] connected {self.port}"); return True
        except Exception as e:
            print("[SERIAL] open error:", e); self.ser=None; return False
    def _raw_send(self, line):
        if not self.ser or not self.ser.is_open:
            if not self._connect_serial(): return
        if not line.endswith("\r\n"): line+="\r\n"
        if Cfg.DEBUG_TX: print("[TX]", line.strip())
        try: self.ser.write(line.encode())
        except Exception as e:
            print("[SERIAL] write error:", e)
            try: self.ser.close()
            finally: self.ser=None
    def _send_angles(self, ax, ay):
        if Cfg.PROTOCOL_MODE.upper()=="XY":
            self._raw_send(f"X:{int(ax)},Y:{int(ay)}")
            if Cfg.SEND_Y_TWICE:
                # дублируем Y отдельной строкой — для «капризных» парсеров
                self._raw_send(f"Y:{int(ay)}")
        else:
            self._raw_send(f"ANG {ax-90:.1f} {ay-90:.1f}")
    def _send_laser(self, on):
        self.laser_on=on
        self._raw_send("LAS 1" if on else "LAS 0")
        if Cfg.SEND_ALT_LAS:
            self._raw_send("LAS:1" if on else "LAS:0")
            self._raw_send("LASER 1" if on else "LASER 0")
    def _limit_rate_and_send(self):
        now=time.time()
        if now>=self._next_tx:
            self._send_angles(self.servo_x, self.servo_y)
            self._next_tx=now+1.0/Cfg.CMD_HZ

    # ---------- Calib file ----------
    def _load_calib(self):
        if os.path.exists(Cfg.CALIB_PATH):
            try:
                with open(Cfg.CALIB_PATH,"r") as f:
                    d=json.load(f)
                    self.invert_x=bool(d.get("invert_x", self.invert_x))
                    self.invert_y=bool(d.get("invert_y", self.invert_y))
                    self.swap_axes=bool(d.get("swap_axes", self.swap_axes))
                    self.deg_per_px_x=d.get("deg_per_px_x", self.deg_per_px_x)
                    self.deg_per_px_y=d.get("deg_per_px_y", self.deg_per_px_y)
                    self.kp=float(d.get("kp", self.kp)); self.kd=float(d.get("kd", self.kd))
                    # защита от некорректных масштабов
                    if not (isinstance(self.deg_per_px_x,(int,float)) and 0.02<=self.deg_per_px_x<=0.3):
                        self.deg_per_px_x=None
                    if not (isinstance(self.deg_per_px_y,(int,float)) and 0.02<=self.deg_per_px_y<=0.3):
                        self.deg_per_px_y=None
                    print("[CALIB] loaded", d)
            except Exception as e:
                print("[CALIB] load warn:", e)
    def _save_calib(self):
        d={"invert_x":self.invert_x,"invert_y":self.invert_y,"swap_axes":self.swap_axes,
           "deg_per_px_x":self.deg_per_px_x,"deg_per_px_y":self.deg_per_px_y,
           "kp":self.kp,"kd":self.kd}
        try:
            with open(Cfg.CALIB_PATH,"w") as f: json.dump(d,f,indent=2)
            print("[CALIB] saved", d)
        except Exception as e:
            print("[CALIB] save warn:", e)

    # ---------- Autocalib ----------
    def _avg_center(self, n=Cfg.CALI_SAMPLES):
        acc=[]; tries=0
        while len(acc)<n and tries<n*3:
            frame=self._capture()
            res=self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
            box=select_target(res, self.wanted_ids, None)
            if box is not None: acc.append((box[0], box[1]))
            tries+=1
        if not acc: return None
        mx=sum(x for x,_ in acc)/len(acc); my=sum(y for _,y in acc)/len(acc)
        return (mx,my)
    def run_autocalib(self):
        print("[CALIB] start… (нужна стабильная цель)")
        base=self._avg_center()
        if base is None: print("[CALIB] цели нет"); return False
        bx,by=base
        # X
        d= Cfg.CALI_POKE_DEG if self.servo_x+Cfg.CALI_POKE_DEG<=Cfg.MAX_ANGLE else -Cfg.CALI_POKE_DEG
        self.servo_x=clamp(self.servo_x+d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(Cfg.CALI_WAIT_S)
        a=self._avg_center()
        if a is None: print("[CALIB] X: цель пропала"); return False
        dx=a[0]-bx
        self.invert_x = (dx*d>0)
        if abs(dx)>=Cfg.CALI_MIN_DPX: self.deg_per_px_x=abs(d/dx)
        print(f"[CALIB] X d={d:+.1f}° -> dx={dx:+.1f}px invX={self.invert_x} deg/px_x={self.deg_per_px_x}")
        self.servo_x=clamp(self.servo_x-d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); time.sleep(0.05)
        # Y
        d= Cfg.CALI_POKE_DEG if self.servo_y+Cfg.CALI_POKE_DEG<=Cfg.MAX_ANGLE else -Cfg.CALI_POKE_DEG
        self.servo_y=clamp(self.servo_y+d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send()
        time.sleep(Cfg.CALI_WAIT_S)
        a=self._avg_center()
        if a is None: print("[CALIB] Y: цель пропала"); return False
        dy=a[1]-by
        self.invert_y = (dy*d>0)
        if abs(dy)>=Cfg.CALI_MIN_DPX: self.deg_per_px_y=abs(d/dy)
        print(f"[CALIB] Y d={d:+.1f}° -> dy={dy:+.1f}px invY={self.invert_y} deg/px_y={self.deg_per_px_y}")
        self.servo_y=clamp(self.servo_y-d, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); time.sleep(0.05)
        # валидация масштабов
        if not (self.deg_per_px_x and 0.02<=self.deg_per_px_x<=0.3): self.deg_per_px_x=None
        if not (self.deg_per_px_y and 0.02<=self.deg_per_px_y<=0.3): self.deg_per_px_y=None
        self._save_calib(); print("[CALIB] done"); return True

    # ---------- Управление ----------
    def _px_to_deg(self, dx, dy):
        # запасной вариант: от FOV (Cam v3 ~66° x 41°)
        dppx = self.deg_per_px_x if self.deg_per_px_x else (66.0/self.w)
        dppy = self.deg_per_px_y if self.deg_per_px_y else (41.0/self.h)
        # инверсии: хотим уменьшать ошибку
        ex = (-dx if self.invert_x else dx) * dppx
        ey = (-dy if self.invert_y else dy) * dppy
        return ex,ey

    def _pd_step(self, dx_px, dy_px):
        if abs(dx_px)<Cfg.DEAD_PX: dx_px=0
        if abs(dy_px)<Cfg.DEAD_PX: dy_px=0

        # Перестановка осей (если Arduino/wiring поменяли места X и Y)
        if self.swap_axes:
            dx_px, dy_px = dy_px, dx_px

        ex,ey = self._px_to_deg(dx_px, dy_px)
        now=time.time(); dt=max(0.001, now-self.prev_t)
        d_ex=(ex-self.prev_err_deg_x)/dt; d_ey=(ey-self.prev_err_deg_y)/dt
        self.prev_err_deg_x, self.prev_err_deg_y, self.prev_t = ex,ey,now

        step_x=clamp(self.kp*ex + self.kd*d_ex, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)
        step_y=clamp(self.kp*ey + self.kd*d_ey, -Cfg.MAX_STEP_DEG, Cfg.MAX_STEP_DEG)

        # если оси свопнуты — применяем соответственно
        if self.swap_axes:
            self.servo_y = clamp(self.servo_y + step_x, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
            self.servo_x = clamp(self.servo_x + step_y, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
        else:
            self.servo_x = clamp(self.servo_x + step_x, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)
            self.servo_y = clamp(self.servo_y + step_y, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE)

        self._limit_rate_and_send()

    def _scan_step(self):
        self.servo_x += self.scan_dir*Cfg.SCAN_SPEED_DEG
        if self.servo_x>=Cfg.SCAN_MAX_ANGLE:
            self.servo_x=Cfg.SCAN_MAX_ANGLE; self.scan_dir=-1
        elif self.servo_x<=Cfg.SCAN_MIN_ANGLE:
            self.servo_x=Cfg.SCAN_MIN_ANGLE; self.scan_dir=1
        self._limit_rate_and_send()

    # ---------- Main loop ----------
    def run(self):
        print("Q=выход | I/K — инверт X/Y | X — поменять оси местами | R — автокалибровка | U/J — Kp +/- | F/D — Kd +/- | L — лазер | 1/2/3/4/0 — тест серво")
        try:
            while True:
                frame=self._capture()
                results=self.model.predict(frame, conf=Cfg.CONF_THRESHOLD, imgsz=Cfg.IMGSZ, verbose=False)
                box=select_target(results, self.wanted_ids,
                                  (self.last_box[0], self.last_box[1]) if self.last_box else None)

                if box is not None and self.last_box is not None:
                    # отсечём дикие скачки
                    if abs(box[0]-self.last_box[0])+abs(box[1]-self.last_box[1])>Cfg.MAX_JUMP_PX:
                        box=None

                if box is not None:
                    self.last_box=box; self.keep_frames=0; self.missing_frames_real=0
                    dx= box[0]-self.cx; dy= box[1]-self.cy
                    self._pd_step(dx,dy)
                    # гистерезис лазера
                    err=max(abs(dx),abs(dy))
                    if err<=Cfg.LOCK_ERR_PX:
                        self.lock_ok+=1; self.lock_bad=0
                    else:
                        self.lock_ok=max(0,self.lock_ok-1); self.lock_bad+=1
                    if (not self.laser_on) and self.lock_ok>=Cfg.LOCK_ON_FR: self._send_laser(True)
                    if self.laser_on and self.lock_bad>=Cfg.LOCK_OFF_FR: self._send_laser(False)
                else:
                    self.missing_frames_real+=1
                    if self.last_box is not None and self.keep_frames<Cfg.BOX_KEEP_FR:
                        self.keep_frames+=1
                    else:
                        self.last_box=None
                        if self.missing_frames_real>=Cfg.MISS_FOR_SCAN: self._scan_step()
                        else:
                            self.servo_x += (90 - self.servo_x)*0.05
                            self.servo_y += (90 - self.servo_y)*0.05
                            self._limit_rate_and_send()
                    self.lock_ok=0; self.lock_bad+=1
                    if self.laser_on and self.lock_bad>=Cfg.LOCK_OFF_FR: self._send_laser(False)

                # overlay
                cv2.drawMarker(frame,(self.cx,self.cy),(255,255,255),cv2.MARKER_CROSS,20,2)
                if self.last_box:
                    cx,cy,x1,y1,x2,y2=self.last_box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,200,0),2)
                    cv2.circle(frame,(cx,cy),5,(255,0,0),2)
                txt=(f"Kp:{self.kp:.2f} Kd:{self.kd:.2f} invX:{int(self.invert_x)} invY:{int(self.invert_y)} "
                     f"swap:{int(self.swap_axes)} Laser:{'ON' if self.laser_on else 'off'} Port:{self.port or '-'}")
                cv2.putText(frame, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0),2)
                cv2.imshow("Auto Aim — robust (axis-fix + dup-cmds)", frame)

                key=cv2.waitKey(1)&0xFF
                if key==ord('q'): break
                elif key==ord('i'): self.invert_x=not self.invert_x; print("invert_x:",self.invert_x)
                elif key==ord('k'): self.invert_y=not self.invert_y; print("invert_y:",self.invert_y)
                elif key==ord('x'): self.swap_axes=not self.swap_axes; print("swap_axes:",self.swap_axes)
                elif key==ord('r'): self.run_autocalib()
                elif key==ord('u'): self.kp=min(0.6, self.kp+0.02); print("kp:",self.kp)
                elif key==ord('j'): self.kp=max(0.12, self.kp-0.02); print("kp:",self.kp)
                elif key==ord('f'): self.kd+=0.03; print("kd:",self.kd)
                elif key==ord('d'): self.kd=max(0.0,self.kd-0.03); print("kd:",self.kd)
                elif key==ord('l'): self._send_laser(not self.laser_on)
                # самотест сервоприводов: мгновенно видно, доходят ли команды до Arduino
                elif key==ord('1'): self.servo_x=clamp(self.servo_x-10, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); print("X -10")
                elif key==ord('2'): self.servo_x=clamp(self.servo_x+10, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); print("X +10")
                elif key==ord('3'): self.servo_y=clamp(self.servo_y-10, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); print("Y -10")
                elif key==ord('4'): self.servo_y=clamp(self.servo_y+10, Cfg.MIN_ANGLE, Cfg.MAX_ANGLE); self._limit_rate_and_send(); print("Y +10")
                elif key==ord('0'): self.servo_x=90; self.servo_y=90; self._limit_rate_and_send(); print("center 90/90")
                elif key==ord('s'): self._save_calib()

        except Exception as e:
            print("[FATAL]", e); traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        print("\n[EXIT] shutting down…")
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
