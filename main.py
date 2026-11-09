import os
import sys
import csv
import time
import json
import pickle
import argparse
import threading
import re
import glob
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
from scipy.optimize import differential_evolution, nnls

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================================
# 0. ЦЕНТРАЛІЗОВАНА КОНФІГУРАЦІЯ
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).resolve().parent
    AI_DIR = BASE_DIR / "ai"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_FILE = AI_DIR / "config.json"
    MODEL_FILE = AI_DIR / "tf_model.keras"
    SCALER_FILE = AI_DIR / "tf_scaler.pkl"

    BAUD_RATE = 115200
    # Рівно 7 полів: Current_A, Voltage, Omega, Target, PWM, Ticks, dTicks.
    TELEMETRY_REGEX = re.compile(
        r'^-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){6}$'
    )

    BOUNDS_KP = (0.1, 1.5)
    BOUNDS_KI = (0.01, 1.0)
    BOUNDS_KD = (0.0001, 0.02)
    PENALTY_OVERSHOOT = 500.0
    MAX_TARGET_ABS = 21.0
    PWM_DEADZONE = 12.0

    DEFAULT_MOTOR_PARAMS = {
        "R": 4.5, "L": 0.01, "Ke": 0.6, "Km": 0.6,
        "J": 0.001, "b": 1e-06, "Ts": 0.05, "Rd": 0.05
    }

    @classmethod
    def ensure_dirs(cls):
        cls.AI_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_calibrated_pbc_params(cls):
        try:
            with open(cls.CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            m = cfg['motor_parameters']
            return m.get('R', 4.5), m.get('Ke', 0.6), m.get('Rd', 0.05)
        except Exception:
            return 4.5, 0.6, 0.05

# ============================================================================
# 1. ЦИФРОВИЙ ДВІЙНИК (СИНХРОНІЗОВАНИЙ З ПРОШИВКОЮ)
# ============================================================================
class DigitalTwin:
    def __init__(self, config_path: Path = Config.CONFIG_FILE):
        self.config = self._load_config(config_path)
        m = self.config['motor_parameters']
        self.R, self.L, self.Ke = m.get('R', 4.5), m.get('L', 0.01), m.get('Ke', 0.6)
        self.Km, self.J, self.b, self.Ts = (
            m.get('Km', 0.6), m.get('J', 0.001),
            m.get('b', 1e-6), m.get('Ts', 0.05)
        )
        self.Rd = m.get('Rd', 0.05)
        self.V_supply = float(self.config.get('hardware', {}).get('supply_voltage', 12.0))
        self.PWM_DEADZONE = Config.PWM_DEADZONE

    def _load_config(self, path: Path) -> Dict:
        if not path.exists():
            return {"hardware": {"supply_voltage": 12.0},
                    "motor_parameters": Config.DEFAULT_MOTOR_PARAMS.copy()}
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data.setdefault("hardware", {})
        data.setdefault("motor_parameters", {})
        merged = Config.DEFAULT_MOTOR_PARAMS.copy()
        merged.update(data["motor_parameters"])
        data["motor_parameters"] = merged
        return data

    def simulate_closed_loop(self, target: float, kp: float, ki: float, kd: float, x0: float = 0.0, dt: float = 0.01, T: float = 4.0) -> Tuple[float, float]:
        steps = int(T / dt)
        i, w = 0.0, x0
        e_sum, prev_e, d_filtered = 0.0, target - w, 0.0
        iae, peak_progress = 0.0, (1.0 if target >= 0 else -1.0) * w
        current_pwm = 0.0
        max_internal_dt = 0.001
        
        for _ in range(steps):
            e = target - w
            # Інтегральне вікно (як у прошивці)
            if abs(e) < 6.0:
                e_sum = np.clip(e_sum + ki * e * dt, -2.0, 2.0)
            else:
                e_sum *= np.exp(-dt / 0.2)
            
            # D-фільтр
            de_raw = (e - prev_e) / dt
            prev_e = e
            d_filtered += (1.0 - np.exp(-dt / 0.02)) * (de_raw - d_filtered)
            
            i_ref = np.clip(e_sum + kp * e + kd * d_filtered, -2.0, 2.0)
            # Та сама формула PBC, що й у прошивці Arduino:
            # u = ke*omega* + R*i_ref - Rd*(i - i_ref).
            u_raw = self.Ke * target + self.R * i_ref - self.Rd * (i - i_ref)
            if target > 0.1 and u_raw > 0: u_raw += 0.3 + 0.02 * abs(target)
            elif target < -0.1 and u_raw < 0: u_raw -= 0.3 + 0.02 * abs(target)

            # Обмеження напруги
            u_raw = np.clip(u_raw, -self.V_supply, self.V_supply)

            # Перетворення в PWM з насиченням
            target_pwm = np.clip(abs(u_raw) / self.V_supply, 0.0, 1.0) * 255.0

            # Slew Rate
            signed_target_pwm = np.sign(u_raw) * target_pwm
            max_delta = 1000.0 * dt
            if signed_target_pwm > current_pwm + max_delta:
                current_pwm += max_delta
            elif signed_target_pwm < current_pwm - max_delta:
                current_pwm -= max_delta
            else:
                current_pwm = signed_target_pwm

            # Deadzone: |PWM| <= 12 -> 0 В, інакше ефективна частина зі знаком
            pwm_mag = abs(current_pwm)
            if pwm_mag <= self.PWM_DEADZONE:
                u = 0.0
            else:
                sign_pwm = 1.0 if current_pwm > 0 else -1.0
                effective_pwm = sign_pwm * (pwm_mag - self.PWM_DEADZONE)
                u = (effective_pwm / 255.0) * self.V_supply
                        
            # Інтеграція з підкроками
            n_sub = max(1, int(np.ceil(dt / max_internal_dt)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                tau_e = self.L / max(self.R, 1e-9)
                i_next = ((u - self.Ke * w) / max(self.R, 1e-9)) + (i - ((u - self.Ke * w) / max(self.R, 1e-9))) * np.exp(-sub_dt / tau_e)
                torque = self.Km * i_next
                
                # Поріг stick-slip відповідає документації: |omega| < 0.1 рад/с.
                if abs(w) < 0.1 and abs(torque) < self.Ts:
                    dw = 0.0
                else:
                    friction = self.Ts * np.sign(torque if abs(w) < 1e-4 else w)
                    dw = (torque - friction - self.b * w) / max(self.J, 1e-12)
                
                i = i_next
                w += dw * sub_dt
                if np.isnan(w) or abs(w) > 200: w = 0.0
                if np.isnan(i) or abs(i) > 10: i = 0.0

            peak_progress = max(peak_progress, (1.0 if target >= 0 else -1.0) * w)
            iae += abs(e) * dt
            
        # Симетрична оцінка для прямого та зворотного напрямків.
        overshoot = ((peak_progress - abs(target)) / abs(target)) * 100.0 if abs(target) > 1e-9 else 0.0
        return iae, max(overshoot, 0.0)

# ============================================================================
# 2. SERIAL MANAGER
# ============================================================================
class SerialManager:
    def __init__(self, port: Optional[str] = None, baudrate: int = Config.BAUD_RATE):
        self.port = port or self._autodetect_port()
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self._stop_event = threading.Event()

    @staticmethod
    def _autodetect_port() -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if any(k in p.description.lower() for k in ["arduino", "ch340", "usb", "cp210"]):
                return p.device
        return ports[0].device if ports else None

    def __enter__(self):
        if not self.port:
            raise ConnectionError("❌ Arduino не знайдено. Вкажіть --port COM5.")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            print(f"🔌 Підключено до {self.port}")
            return self
        except serial.SerialException as e:
            raise ConnectionError(f"❌ Помилка відкриття {self.port}: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self.ser and self.ser.is_open:
            try: self.ser.write(b"s\n"); time.sleep(0.1)
            except: pass
            self.ser.close()
            print("🔒 COM-порт закрито.")

    def send(self, cmd: str):
        if self.ser and self.ser.is_open:
            self.ser.write(f"{cmd}\n".encode())

# ============================================================================
# 3. МОДУЛІ СИСТЕМИ
# ============================================================================
class DataCollector:
    @staticmethod
    def run(scenario_file: Optional[str] = None, port: Optional[str] = None, auto_plot: bool = False, tag: str = "base"):
        Config.ensure_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Config.LOGS_DIR / f"{tag}_telemetry_{timestamp}.csv"
        
        try:
            with SerialManager(port) as hw, open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Current_A", "Voltage", "Omega", "Target", "PWM", "Ticks", "dTicks"])
                
                def read_loop():
                    partial = b""
                    while not hw._stop_event.is_set():
                        if hw.ser.in_waiting == 0: time.sleep(0.001); continue
                        raw = partial + hw.ser.readline()
                        if not raw.endswith(b'\n'): partial = raw; continue
                        partial = b""
                        line = raw.decode('utf-8', errors='ignore').strip()
                        
                        if Config.TELEMETRY_REGEX.match(line):
                            data = tuple(float(p) for p in line.split(','))
                            try:
                                writer.writerow([datetime.now().isoformat()] + list(data))
                                f.flush()
                            except ValueError:
                                break
                        elif line.startswith("[") or line.startswith("---") or line.startswith("DIAG"):
                            # Виводимо тільки службові повідомлення, ігноруємо "сміття"
                            print(f"[Arduino]: {line}")

                threading.Thread(target=read_loop, daemon=True).start()
                
                # АВТОМАТИЧНЕ СКИДАННЯ КОЕФІЦІЄНТІВ ДО БАЗОВИХ (для чесного Baseline)
                print("⚙️ Встановлюємо базові коефіцієнти PID (Kp=0.4, Ki=0.1, Kd=0.002)...")
                # Очищаємо буфер від старого сміття
                hw.ser.reset_input_buffer()
                hw.send("kp 0.400")
                hw.send("ki 0.100")
                hw.send("kd 0.002")
                # Синхронізація PBC-параметрів з калібруванням
                R_cal, Ke_cal, Rd_cal = Config.get_calibrated_pbc_params()
                hw.send(f"r {R_cal:.4f}")
                hw.send(f"ke {Ke_cal:.5f}")
                hw.send(f"rd {Rd_cal:.4f}")
                print(f"⚙️ PBC синхронізовано: R={R_cal:.3f} Ом, "f"ke={Ke_cal:.4f}, rd={Rd_cal:.4f}")
                time.sleep(0.1) # Даємо Arduino час застосувати

                if scenario_file:
                    print(f"🚀 Виконання сценарію: {scenario_file}")
                    with open(scenario_file, "r", encoding="utf-8") as f_scr:
                        for line in f_scr:
                            line = line.strip()
                            if not line or line.startswith("#"): continue
                            if line.lower().startswith("wait "):
                                time.sleep(float(line.split(" ")[1])); continue
                            if line.lower() == "stop_all": break
                            if line.lower().startswith("w "):
                                try:
                                    tgt = float(line.split()[1])
                                    tgt = float(np.clip(tgt, -Config.MAX_TARGET_ABS, Config.MAX_TARGET_ABS))
                                    line = f"w {tgt:.3f}"
                                except Exception:
                                    continue
                            hw.send(line); time.sleep(0.05)
                else:
                    print("💡 Ручний режим (Ctrl+C для виходу).")
                    while True: hw.send(input())
        except KeyboardInterrupt:
            print("\n🛑 Зупинка...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Дані збережено у {csv_path}")
            if auto_plot: Analyzer.run(file_path=str(csv_path))

class Calibrator:
    @staticmethod
    def run():
        Config.ensure_dirs()
        all_logs = glob.glob(str(Config.LOGS_DIR / "*telemetry_*.csv"))
        csv_files = [f for f in all_logs
                     if "ai_" not in Path(f).name and "hybrid_" not in Path(f).name]

        if not csv_files:
            print("❌ Немає базових логів для калібрування.")
            return
        latest = max(csv_files, key=os.path.getctime)

        print(f"📋 Калібрування за файлом: {latest}...")
        df = pd.read_csv(latest)
        pwm      = df['PWM'].values.astype(float)
        current  = df['Current_A'].values.astype(float)
        speed    = df['Omega'].values.astype(float)
        voltage  = df['Voltage'].values.astype(float)

        # Оцінка кроку дискретизації
        dt_est = 0.01
        try:
            ts = pd.to_datetime(df['Timestamp'])
            dts = ts.diff().dt.total_seconds().dropna().values
            if len(dts) > 0:
                med = float(np.median(dts))
                if 1e-5 < med < 1.0:
                    dt_est = med
        except Exception:
            pass

        acceleration = np.abs(np.diff(speed, prepend=speed[0])) / dt_est
        steady_fwd = (
            (speed > 0.5) &
            (current > 0.01) &
            (pwm > 20) &
            (acceleration < 2.0) &
            (voltage > 1.0) &
            np.isfinite(voltage) & np.isfinite(current) & np.isfinite(speed)
        )
        n_fwd = int(np.count_nonzero(steady_fwd))
        if n_fwd < 10:
            print(f"⚠️ Мало forward-steady точок ({n_fwd}). "
                  f"Послаблюємо умови.")
            steady_fwd = (
                (speed > 0.1) &
                (pwm > 10) &
                (acceleration < 5.0) &
                (voltage > 0.5) &
                np.isfinite(voltage) & np.isfinite(current) & np.isfinite(speed)
            )
            n_fwd = int(np.count_nonzero(steady_fwd))
            if n_fwd < 5:
                raise RuntimeError(
                    f"Недостатньо forward-точок ({n_fwd}). Потрібен сценарій "
                    f"з розгоном у прямому напрямку. Запустіть спочатку "
                    f"collect на scenario.txt або scenario-stress.txt."
                )

        A = np.column_stack([current[steady_fwd], speed[steady_fwd]])
        b_vec = voltage[steady_fwd]  # u — виміряна напруга мотора

        # Відкидаємо занадто близькі до нуля b_vec (шум)
        mask = b_vec > 0.01
        if mask.sum() < 5:
            raise RuntimeError("b_vec містить занадто мало додатних значень.")
        A, b_vec = A[mask], b_vec[mask]

        from scipy.optimize import minimize

        def _objective(x):
            R, Ke = x
            return float(np.sum((R * A[:, 0] + Ke * A[:, 1] - b_vec) ** 2))

        res = minimize(
            _objective,
            x0=[5.0, 0.5],
            bounds=[(2.0, 30.0), (0.15, 0.80)],
            method="L-BFGS-B",
        )
        R_cal, Ke_cal = float(res.x[0]), float(res.x[1])
        if R_cal < 2.0:
            print(f"⚠️ R={R_cal:.3f} < 2 Ом — підозріло. Перевірте, "
            f"що INA219 вимірює напругу в правильній точці.")
        residual_l2 = float(np.sqrt(max(res.fun, 0.0)))
        print(f"⚡ Електрика: R={R_cal:.4f} Ом, Ke={Ke_cal:.5f} "
      f"(residual={residual_l2:.3f}, точок={len(b_vec)})")

        pwm_max = float(pwm[steady_fwd].max()) if n_fwd > 0 else float(pwm.max())

        if pwm_max >= 220:
            high_mask = (pwm > 220) & (speed > 1.0) & np.isfinite(voltage)
            if high_mask.sum() >= 3:
                v_supply = round(float(voltage[high_mask].max()), 3)
            else:
                v_supply = round(float(np.nanmax(voltage[steady_fwd])), 3)
            print(f"⚡ V_supply (з PWM≥220): {v_supply} В")
        else:
            # Лінійна регресія на forward-steady: u = a*pwm + c
            from scipy.stats import linregress
            x = pwm[steady_fwd]
            y = voltage[steady_fwd]
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0.5)
            if ok.sum() >= 10:
                slope, intercept, _, _, _ = linregress(x[ok], y[ok])
                v_supply = round(float(slope * 255.0 + intercept), 3)
                print(f"⚡ V_supply (екстраполяція з PWM_max={pwm_max:.0f}): "
                      f"{v_supply} В  [похибка через L298N drop, "
                      f"краще w=21]")
            else:
                v_supply = 12.0
                print(f"⚠️ Замало точок для регресії, V_supply = 12 В (номінал).")

        if v_supply < 1.0:
            v_supply = 12.0
        # Захист від явно неправильних значень
        if v_supply > 14.0:
            print(f"⚠️ V_supply={v_supply} В > 14 В — обмежено 14 В.")
            v_supply = 14.0

        try:
            n = len(current)
            if n < 50:
                raise ValueError(f"мало даних ({n})")

            acc = np.gradient(speed, dt_est)
            valid = (
                (np.abs(speed) > 0.5) &
                (np.abs(current) > 0.01) &
                (np.abs(current) < 3.0) &
                (np.abs(acc) < 100.0) &
                np.isfinite(acc) & np.isfinite(current) & np.isfinite(speed)
            )
            n_valid = int(np.count_nonzero(valid))
            if n_valid < 30:
                raise ValueError(f"мало валідних точок ({n_valid})")

            # Діагностика: чи струм знаковий?
            # Якщо при від'ємній швидкості струм переважно додатний —
            # PowerMonitor інвертує за dir, механіка буде недостовірною.
            neg_speed_mask = speed[valid] < -0.5
            if neg_speed_mask.sum() > 0:
                mean_neg_i = float(np.mean(current[valid][neg_speed_mask]))
                if mean_neg_i > 0.05:
                    print(f"⚠️ Струм при реверсі додатний "
                          f"(mean={mean_neg_i:.3f}). "
                          f"PowerMonitor інвертує за dir — "
                          f"виправте його для достовірної механіки.")

            A_mech = np.column_stack([
                acc[valid],
                np.sign(speed[valid]),
                speed[valid]
            ])
            b_mech = Ke_cal * current[valid]

            cond_num = float(np.linalg.cond(A_mech))
            if cond_num > 1e6:
                print(f"⚠️ Погана обумовленість механічної матриці "
                      f"(cond={cond_num:.1e}) — дефолти.")
                J_cal, b_cal, Ts_cal = 0.001, 1e-06, 0.05
            else:
                (J_cal, Ts_cal, b_cal), _ = nnls(A_mech, b_mech)
                J_cal  = float(np.clip(J_cal,  1e-5, 0.1))
                b_cal  = float(np.clip(b_cal,  1e-8, 0.01))
                Ts_cal = float(np.clip(Ts_cal, 0.0,  0.5))
                print(f"⚙️ Механіка: J={J_cal:.6f}, b={b_cal:.2e}, "
                      f"Ts={Ts_cal:.4f} (cond={cond_num:.1e})")
        except Exception as e:
            print(f"⚠️ Механічна ідентифікація пропущена: {e}")
            J_cal, b_cal, Ts_cal = 0.001, 1e-06, 0.05

        params = {
            "hardware": {"supply_voltage": v_supply},
            "motor_parameters": {
                "R":  round(R_cal, 4), "L": 0.01,
                "Ke": round(Ke_cal, 5), "Km": round(Ke_cal, 5),
                "J":  round(J_cal, 6),  "b":  round(b_cal, 8),
                "Ts": round(Ts_cal, 4), "Rd": 0.05
            }
        }
        with open(Config.CONFIG_FILE, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"✅ Двійник відкалібровано: "
              f"R={R_cal:.4f} Ом, Ke={Ke_cal:.5f}, "
              f"V_supply={v_supply} В, "
              f"J={J_cal:.6f}, b={b_cal:.2e}, Ts={Ts_cal:.4f}")

def _de_objective(g, twin, target, penalty):
    kp, ki, kd = g
    T_sim = 4.0 if target > 15 else (3.0 if target > 8 else 2.5)   # ← додати
    total = 0.0
    for frac in (0.0, 0.5):
        x0 = target * frac
        iae, over = twin.simulate_closed_loop(
            target, kp, ki, kd, x0=x0, T=T_sim                         # ← T=T_sim
        )
        pen = (over - 15.0) * penalty if over > 15.0 else 0.0
        total += iae + pen
    return total / 2.0

class AITrainer:
    @staticmethod
    def run(max_target: float = 21.0):
        Config.ensure_dirs()
        import tensorflow as tf
        from tensorflow.keras import layers, models

        np.random.seed(42)
        import random as _random
        _random.seed(42)
        tf.random.set_seed(42)
        
        print(f"🧠 Генерація даних (від 3.0 до {max_target} рад/с) та навчання ШІ...")
        twin = DigitalTwin()
        bounds = [Config.BOUNDS_KP, Config.BOUNDS_KI, Config.BOUNDS_KD]
        X, Y = [], []
        
        if not 3.0 <= max_target <= 21.0:
            raise ValueError("max_target має бути в діапазоні 3..21 рад/с згідно з документацією.")
        targets = np.linspace(3.0, max_target, 13) 
        
        for i, target in enumerate(targets):
            print(f"  Оптимізація для {target:.1f} рад/с ({i+1}/{len(targets)})")
            from functools import partial

            T_sim = 4.0 if target > 15 else (3.0 if target > 8 else 2.5)  # п.3
            obj = partial(_de_objective, twin=twin, target=target,
                        penalty=Config.PENALTY_OVERSHOOT)

            res = differential_evolution(
                obj, bounds,
                maxiter=7, popsize=6, seed=42+i,
                workers=-1, updating='deferred',
            )

            # Додаємо різні початкові швидкості (x0) для наукової новизни
            for _ in range(60):
                # Навчальні приклади охоплюють як розгін, так і гальмування
                # (від'ємну похибку), а не лише початкову позитивну похибку.
                x0 = np.random.uniform(0.0, target * 1.25)
                err = target - x0
                X.append([target, err])
                Y.append(res.x)
        
        # FIX: спочатку поділ за цілями, потім нормалізація на train
        X_arr = np.array(X)  # [target, error]
        Y_arr = np.array(Y)

        # Унікальні цілі
        unique_targets = np.unique(X_arr[:, 0])
        rng = np.random.default_rng(42)
        rng.shuffle(unique_targets)
        n_val = max(1, int(len(unique_targets) * 0.2))
        val_targets = unique_targets[-n_val:]

        val_mask = np.isin(X_arr[:, 0], val_targets)
        train_mask = ~val_mask

        X_tr, X_va = X_arr[train_mask], X_arr[val_mask]
        Y_tr, Y_va = Y_arr[train_mask], Y_arr[val_mask]

        # Нормалізація ТІЛЬКИ на train
        X_m, X_s = X_tr.mean(0), X_tr.std(0)
        Y_m, Y_s = Y_tr.mean(0), Y_tr.std(0)

        X_n = (X_arr - X_m) / np.where(X_s == 0, 1, X_s)
        Y_n = (Y_arr - Y_m) / np.where(Y_s == 0, 1, Y_s)

        X_train = (X_tr - X_m) / np.where(X_s == 0, 1, X_s)
        Y_train = (Y_tr - Y_m) / np.where(Y_s == 0, 1, Y_s)
        X_val = (X_va - X_m) / np.where(X_s == 0, 1, X_s)
        Y_val = (Y_va - Y_m) / np.where(Y_s == 0, 1, Y_s)
        
        model = models.Sequential([
            layers.Input((2,)), layers.Dense(32, activation='relu'), 
            layers.Dense(16, activation='relu'), layers.Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=16, verbose=1)
        
        model.save(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'wb') as f:
            pickle.dump({'X_mean': X_m, 'X_std': X_s, 'Y_mean': Y_m, 'Y_std': Y_s}, f)
        
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Історія навчання (з валідацією)')
        plt.legend(); plt.grid(True)
        plt.savefig(Config.LOGS_DIR / "training_history.png", dpi=150)
        plt.close()
        print("✅ Модель збережена!")

class HILController:
    @staticmethod
    def run(target_speed: float, port: Optional[str] = None, auto_plot: bool = False, static_mode: bool = False, tag: str = "ai", scenario_file: Optional[str] = None):
        Config.ensure_dirs()
        import tensorflow as tf

        target_speed = float(np.clip(target_speed, -Config.MAX_TARGET_ABS, Config.MAX_TARGET_ABS))
        
        if not Config.MODEL_FILE.exists(): 
            print("❌ Модель не знайдена."); return
            
        model = tf.keras.models.load_model(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
        
        mode_name = "Hybrid (Static AI)" if static_mode else "Dynamic AI"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Config.LOGS_DIR / f"{tag}_telemetry_{timestamp}.csv"
        
        # Прапорець для автоматичного завершення
        scenario_done = threading.Event()
        
        try:
            with SerialManager(port) as hw, open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Current_A", "Voltage", "Omega", "Target", "PWM", "Kp", "Ki", "Kd"])
                
                print(f"🚀 HIL-тест ({mode_name}): Ціль: {target_speed} рад/с")
                R_cal, Ke_cal, Rd_cal = Config.get_calibrated_pbc_params()
                hw.send(f"r {R_cal:.4f}")
                hw.send(f"ke {Ke_cal:.5f}")
                hw.send(f"rd {Rd_cal:.4f}")
                time.sleep(0.1)
                hw.send("cfreq 100")
                
                def run_scenario():
                    if not os.path.exists(scenario_file):
                        print(f"❌ Файл {scenario_file} не знайдено!")
                        scenario_done.set()
                        return
                        
                    with open(scenario_file, "r", encoding="utf-8") as f_scr:
                        lines = [l.strip() for l in f_scr if l.strip() and not l.strip().startswith("#")]
                        
                    if not lines:
                        print("⚠️ Сценарій порожній!")
                        scenario_done.set()
                        return
                        
                    print(f"📜 Виконуємо {len(lines)} команд...")
                    for line in lines:
                        if line.lower().startswith("wait"):
                            try:
                                delay = float(line.split()[1])
                                print(f"⏳ Пауза {delay} сек...")
                                time.sleep(delay)
                            except Exception as e: 
                                print(f"⚠️ Помилка паузи: {e}")
                            continue
                            
                        if line.lower() == "stop_all":
                            print("🏁 Знайдено stop_all.")
                            break

                        if line.lower().startswith("w "):
                            try:
                                tgt = float(line.split()[1])
                                tgt = float(np.clip(tgt,
                                                    -Config.MAX_TARGET_ABS,
                                                    Config.MAX_TARGET_ABS))
                                line = f"w {tgt:.3f}"
                            except Exception as e:
                                print(f"⚠️ Некоректна w-команда '{line}': {e}")
                                continue

                        print(f"➡️ Відправка: {line}")
                        hw.send(line)
                        time.sleep(0.05)
                    
                    time.sleep(2) # Даємо час на останні логи
                    scenario_done.set() 
                    print("\n🏁 Сценарій завершено. Зупинка HIL-тесту...")

                if scenario_file:
                    threading.Thread(target=run_scenario, daemon=True).start()
                else:
                    hw.send(f"w {target_speed}")
                
                last_tune, kp, ki, kd = time.time(), 0.4, 0.1, 0.002
                is_calibrated = False
                
                while not scenario_done.is_set():
                    line = hw.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) == 7:
                            curr, volt, omega, target, pwm = map(float, parts[:5])
                            
                            need_update = (not static_mode and time.time() - last_tune > 0.1) or (static_mode and not is_calibrated)
                            
                            if need_update and target != 0:
                                try:
                                    error = target - omega
                                    # Симетричне перетворення для моделі, навченої на
                                    # додатних швидкостях; знак похибки зберігається.
                                    direction = 1.0 if target >= 0 else -1.0
                                    nn_target = np.clip(abs(target), 3.0, 21.0)
                                    nn_error = np.clip(direction * error, -6.0, 6.0)
                                    
                                    X = np.array([[nn_target, nn_error]])
                                    X_n = (X - scaler['X_mean']) / np.where(scaler['X_std'] == 0, 1, scaler['X_std'])
                                    Y = model.predict(X_n, verbose=0) * scaler['Y_std'] + scaler['Y_mean']
                                    
                                    kp, ki, kd = np.clip(Y[0], [Config.BOUNDS_KP[0], Config.BOUNDS_KI[0], Config.BOUNDS_KD[0]], [Config.BOUNDS_KP[1], Config.BOUNDS_KI[1], Config.BOUNDS_KD[1]])
                                    
                                    if not np.all(np.isfinite([kp, ki, kd])):
                                        raise ValueError("AI повернула NaN або нескінченне значення")
                                        
                                    hw.send(f"kp {kp:.3f}"); hw.send(f"ki {ki:.3f}"); hw.send(f"kd {kd:.5f}")
                                    print(f"[{mode_name}] Target: {target:4.1f} | Error: {error:5.2f} | Kp={kp:.3f}, Ki={ki:.3f}")
                                    last_tune = time.time()
                                    is_calibrated = True
                                except Exception as ai_error:
                                    print(f"⚠️ AI Error: {ai_error}")
                            
                            try:
                                writer.writerow([datetime.now().isoformat(), curr, volt, omega, target, pwm, f"{kp:.4f}", f"{ki:.4f}", f"{kd:.5f}"])
                                f.flush()
                            except ValueError:
                                break
                            
        except KeyboardInterrupt:
            print(f"\n🛑 Ручна зупинка {mode_name}...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Лог {mode_name} збережено у {csv_path}")
            if auto_plot: Analyzer.run(file_path=str(csv_path))
            
# ============================================================================
# 4. АНАЛІЗ ТА ЗВІТ
# ============================================================================
class Analyzer:
    @staticmethod
    def run(file_path: Optional[str] = None, save_path: str = "logs/final_ai_result.png"):
        if not file_path:
            files = glob.glob(str(Config.LOGS_DIR / "*telemetry*.csv"))
            if not files: print("❌ Немає логів."); return
            file_path = max(files, key=os.path.getctime)
            
        df = pd.read_csv(file_path)
        t = (pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['Timestamp']).iloc[0]).dt.total_seconds().values
        plt.style.use('seaborn-v0_8-whitegrid')
        
        has_gains = 'Kp' in df.columns
        fig, axs = plt.subplots(4 if has_gains else 3, 1, figsize=(12, 14 if has_gains else 10), sharex=True)
        
        axs[0].plot(t, df['Omega'], 'b-', linewidth=2, label='Фактична')
        axs[0].plot(t, df['Target'], 'r--', linewidth=1.5, label='Цільова')
        axs[0].set_ylabel('Швидкість (рад/с)'); axs[0].legend(); axs[0].grid(True)
        
        axs[1].plot(t, df['PWM'], 'g-', label='PWM'); axs[1].set_ylabel('ШІМ'); axs[1].legend(); axs[1].grid(True)
        
        axs[2].plot(t, df['Current_A'], 'm-', label='Струм'); axs[2].set_ylabel('Струм (А)'); axs[2].grid(True)
        if not has_gains: axs[2].set_xlabel('Час (с)'); axs[2].legend()
        
        if has_gains:
            axs[3].plot(t, df['Kp'], 'c-', label='Kp')
            axs[3].plot(t, df['Ki'], 'm-', label='Ki')
            axs[3].plot(t, df['Kd'] * 100, 'y--', label='Kd × 100')
            axs[3].set_ylabel('Коефіцієнти'); axs[3].set_xlabel('Час (с)'); axs[3].legend(); axs[3].grid(True)
        
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(output, dpi=150); plt.close()
        print(f"✅ Графік збережено: {output}")

    @staticmethod
    def compare(file1: str, file2: str, save_path: str = "logs/comparison.png"):
        df_b, df_a = pd.read_csv(file1), pd.read_csv(file2)
        t_b = (pd.to_datetime(df_b['Timestamp']) - pd.to_datetime(df_b['Timestamp']).iloc[0]).dt.total_seconds().values
        t_a = (pd.to_datetime(df_a['Timestamp']) - pd.to_datetime(df_a['Timestamp']).iloc[0]).dt.total_seconds().values
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(t_b, df_b['Omega'], 'b-', alpha=0.7, label='Baseline')
        ax1.plot(t_a, df_a['Omega'], 'r-', label='AI')
        ax1.plot(t_b, df_b['Target'], 'k--', label='Target')
        ax1.set_ylabel('Швидкість (рад/с)'); ax1.legend(); ax1.grid(True)
        
        ax2.plot(t_b, df_b['PWM'], 'b-', alpha=0.7, label='Baseline')
        ax2.plot(t_a, df_a['PWM'], 'g-', label='AI')
        ax2.set_ylabel('ШІМ'); ax2.set_xlabel('Час (с)'); ax2.legend(); ax2.grid(True)
        
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(output, dpi=150); plt.close()
        print(f"✅ Порівняльний графік збережено: {output}")

class ReportGenerator:
    @staticmethod
    def _overshoot_by_transitions(target, omega,
                                threshold=0.5, min_samples=20):
        target = np.asarray(target, dtype=float)
        omega = np.asarray(omega, dtype=float)
        if len(target) < 2:
            return 0.0

        diffs = np.abs(np.diff(target))
        change_idx = np.where(diffs > threshold)[0]

        # Одна цільова точка (плоский Target)
        if len(change_idx) == 0:
            tgt = target[-1]
            if abs(tgt) < 1e-6:
                return 0.0
            if tgt > 0:
                return float(max(0.0, (omega.max() - tgt) / abs(tgt) * 100.0))
            else:
                return float(max(0.0, (tgt - omega.min()) / abs(tgt) * 100.0))

        boundaries = [0] + [c + 1 for c in change_idx] + [len(target)]

        max_os = 0.0
        for k in range(len(boundaries) - 1):
            start = boundaries[k]
            end = boundaries[k + 1]
            if end - start < min_samples:
                continue

            new_tgt = target[end - 1]
            if abs(new_tgt) < 1e-6:
                continue

            # old_tgt — ціль ДО переходу (для першого сегмента — 0)
            if start > 0:
                old_tgt = float(target[start - 1])
            else:
                old_tgt = 0.0

            seg = omega[start:end]

            if new_tgt > old_tgt + 0.1:
                peak = float(seg.max())
                os_val = max(0.0, (peak - new_tgt) / abs(new_tgt) * 100.0)
            elif new_tgt < old_tgt - 0.1:
                valley = float(seg.min())
                os_val = max(0.0, (new_tgt - valley) / abs(new_tgt) * 100.0)
            else:
                os_val = 0.0

            max_os = max(max_os, os_val)
        return float(max_os)

    @staticmethod
    def create_multi_scenario_pdf(data_list: list, save_path: str = "logs/Final_Multi_Report.pdf"):
        import platform
        from fpdf import FPDF 
        
        Config.ensure_dirs()
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pdf = FPDF()
        font_path = "C:/Windows/Fonts/arial.ttf" if platform.system() == "Windows" else "DejaVuSans.ttf"
        
        pdf.add_page()
        try:
            pdf.add_font("Arial", "", font_path, uni=True)
            pdf.set_font("Arial", size=20)
        except:
            pdf.set_font("Helvetica", size=20)
            
        pdf.ln(60)
        pdf.cell(200, 20, text="ЗВІТ ПРО РЕЗУЛЬТАТИ HIL-ДОСЛІДЖЕННЯ", new_x="LMARGIN", new_y="NEXT", align='C')
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, text="Порівняння Baseline, Hybrid та Dynamic AI", new_x="LMARGIN", new_y="NEXT", align='C')

        def crop_active_phase(df):
            if len(df) == 0 or 'Timestamp' not in df.columns:
                return df.copy().reset_index(drop=True), np.array([0.0])

            nonzero = df[df['Target'] != 0].index
            if len(nonzero) == 0:
                t = (pd.to_datetime(df['Timestamp'])
                    - pd.to_datetime(df['Timestamp']).iloc[0]
                    ).dt.total_seconds().values
                return df.copy().reset_index(drop=True), t
            start_idx = nonzero[0]
            end_idx = min(nonzero[-1] + 100, len(df) - 1)
            cropped = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            t = (pd.to_datetime(cropped['Timestamp'])
                - pd.to_datetime(cropped['Timestamp']).iloc[0]
                ).dt.total_seconds().values
            return cropped, t
        
        def _fmt(v, prec=2, suffix=""):
            if v is None or not np.isfinite(v):
                return "n/a"
            return f"{v:.{prec}f}{suffix}"
        
        def calc_metrics(df):
            if len(df) == 0 or 'Target' not in df.columns:
                return float('nan'), float('nan'), float('nan')
            cropped, t = crop_active_phase(df)
            if len(cropped) == 0 or len(t) < 2:
                return float('nan'), float('nan'), float('nan')
            error = cropped['Target'].values - cropped['Omega'].values
            iae = np.trapezoid(np.abs(error), x=t)
            if abs(cropped['Target'].max()) < 1e-9:
                return float('nan'), float('nan'), float('nan')

            overshoot = ReportGenerator._overshoot_by_transitions(
                cropped['Target'].values, cropped['Omega'].values)

            # Статична похибка на ОСТАННІЙ сталій ділянці перед зміною цілі
            target_arr = cropped['Target'].values
            diffs = np.abs(np.diff(target_arr))
            changes = np.where(diffs > 0.5)[0]
            if len(changes) == 0:
                seg_start, seg_end = 0, len(target_arr)
            else:
                last_change = changes[-1]
                # від останньої зміни до кінця — це останнє плато (якщо довге)
                if len(target_arr) - last_change > 100:
                    seg_start, seg_end = last_change + 1, len(target_arr)
                else:
                    seg_start, seg_end = changes[-2] + 1 if len(changes) > 1 else 0, last_change
            seg_t = target_arr[seg_start:seg_end]
            if len(seg_t) == 0 or abs(np.median(seg_t)) < 1e-6:
                return iae, overshoot, 0.0
            start_ss = seg_start + int(len(seg_t) * 0.7)
            end_ss = seg_start + int(len(seg_t) * 0.95)
            steady_err = float(np.mean(np.abs(
                cropped['Target'].values[start_ss:end_ss]
                - cropped['Omega'].values[start_ss:end_ss])))
            return iae, overshoot, steady_err

        for idx, data in enumerate(data_list):
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, text=f"Сценарій: {data['name']}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)

            df_b, df_h, df_a = pd.read_csv(data['baseline']), pd.read_csv(data['hybrid']), pd.read_csv(data['ai'])
            
            iae_b, os_b, ss_b = calc_metrics(df_b)
            iae_h, os_h, ss_h = calc_metrics(df_h)
            iae_a, os_a, ss_a = calc_metrics(df_a)

            pdf.set_font("Arial", size=10)
            pdf.cell(50, 10, text="Метрика", border=1, align='C')
            pdf.cell(45, 10, text="Baseline", border=1, align='C')
            pdf.cell(45, 10, text="Hybrid", border=1, align='C')
            pdf.cell(45, 10, text="Dynamic AI", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, text="IAE (Інтеграл)", border=1)
            pdf.cell(45, 10, text=_fmt(iae_b), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(iae_h), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(iae_a), border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, text="Перерегулювання", border=1)
            pdf.cell(45, 10, text=_fmt(os_b, prec=1, suffix="%"), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(os_h, prec=1, suffix="%"), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(os_a, prec=1, suffix="%"), border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, text="Стат. похибка", border=1)
            pdf.cell(45, 10, text=_fmt(ss_b, prec=2), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(ss_h, prec=2), border=1, align='C')
            pdf.cell(45, 10, text=_fmt(ss_a, prec=2), border=1, align='C')
            pdf.ln(15)

            # Малюємо ОБРІЗАНІ графіки
            _, t_b = crop_active_phase(df_b)
            _, t_h = crop_active_phase(df_h)
            _, t_a = crop_active_phase(df_a)
            
            df_b_crop, _ = crop_active_phase(df_b)
            df_h_crop, _ = crop_active_phase(df_h)
            df_a_crop, _ = crop_active_phase(df_a)

            plt.figure(figsize=(10, 4))
            plt.plot(t_b, df_b_crop['Omega'], 'b-', alpha=0.5, label='Baseline')
            plt.plot(t_h, df_h_crop['Omega'], 'orange', linewidth=1.5, label='Hybrid')
            plt.plot(t_a, df_a_crop['Omega'], 'r-', linewidth=2, label='Dynamic AI')
            plt.plot(t_b, df_b_crop['Target'], 'k--', alpha=0.5, label='Target')
            plt.title(f'Динаміка: {data["name"]}'); plt.ylabel('Швидкість (рад/с)'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            
            chart_path = Config.LOGS_DIR / f"temp_chart_{idx}.png"
            plt.savefig(chart_path, dpi=150); plt.close()
            
            pdf.image(str(chart_path), x=15, w=180)
            os.remove(chart_path)

        pdf.output(str(output))
        print(f"✅ Звіт збережено: {output}")

# ============================================================================
# 5. ГОЛОВНИЙ КОНТРОЛЕР
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HIL-комплекс")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_col = subs.add_parser("collect")
    p_col.add_argument("--port", default=None)
    p_col.add_argument("--scenario", default=None)
    p_col.add_argument("--plot", action="store_true")
    p_col.add_argument("--tag", default="base")

    subs.add_parser("calibrate")

    p_train = subs.add_parser("train")
    p_train.add_argument("--max-target", type=float, default=21.0,
                         help="Діапазон навчання 3..21 рад/с згідно з документацією")

    p_ai = subs.add_parser("ai")
    p_ai.add_argument("--port", default=None)
    p_ai.add_argument("--target", type=float, default=15.0)
    p_ai.add_argument("--plot", action="store_true")
    p_ai.add_argument("--tag", default="ai")
    p_ai.add_argument("--static", action="store_true")
    p_ai.add_argument("--scenario", default=None, help="Шлях до сценарію для HIL-тесту")
    
    p_ana = subs.add_parser("analyze")
    p_ana.add_argument("--file", default=None)
    p_ana.add_argument("--save", default="logs/final_ai_result.png")
    
    p_cmp = subs.add_parser("compare")
    p_cmp.add_argument("--file1", required=True)
    p_cmp.add_argument("--file2", required=True)
    p_cmp.add_argument("--save", default="logs/comparison.png")

    subs.add_parser("final-report")
    
    args = parser.parse_args()

    if args.cmd == "collect": DataCollector.run(args.scenario, args.port, args.plot, args.tag)
    elif args.cmd == "calibrate": Calibrator.run()
    elif args.cmd == "train": AITrainer.run(args.max_target)
    elif args.cmd == "ai": HILController.run(args.target, args.port, args.plot, args.static, args.tag, args.scenario)
    elif args.cmd == "analyze": Analyzer.run(args.file, args.save)
    elif args.cmd == "compare": Analyzer.compare(args.file1, args.file2, args.save)
    elif args.cmd == "final-report":
        def get_latest(prefix):
            files = glob.glob(str(Config.LOGS_DIR / f"{prefix}_*.csv"))
            return max(files, key=os.path.getctime) if files else None

        data_list = [
            {
                "name": "Еталонний запуск", 
                "baseline": get_latest("base_scenario"), 
                "hybrid": get_latest("hybrid_scenario"), 
                "ai": get_latest("ai_scenario")
            },
            {
                "name": "Стрес-тест (реверс та гальмування)", 
                "baseline": get_latest("base_stress"), 
                "hybrid": get_latest("hybrid_stress"), 
                "ai": get_latest("ai_stress")
            },
            {
                "name": "Тест частоти (10/100/500 Гц)", 
                "baseline": get_latest("base_freq"), 
                "hybrid": get_latest("hybrid_freq"), 
                "ai": get_latest("ai_freq")
            }
        ]
        data_list = [d for d in data_list if all(d.values())]
        if data_list:
            Config.ensure_dirs()
            ReportGenerator.create_multi_scenario_pdf(data_list, str(Config.LOGS_DIR / "FINAL_THESIS_REPORT.pdf"))
        else: print("❌ Не знайдено повних наборів даних.")