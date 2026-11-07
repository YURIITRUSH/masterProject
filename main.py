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
    TELEMETRY_REGEX = re.compile(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?){6}$') # Рівно 7 полів

    BOUNDS_KP = (0.1, 1.5)
    BOUNDS_KI = (0.01, 1.0)
    BOUNDS_KD = (0.0001, 0.02)
    PENALTY_OVERSHOOT = 500.0

    DEFAULT_MOTOR_PARAMS = {
        "R": 4.5, "L": 0.01, "Ke": 0.6, "Km": 0.6, 
        "J": 0.001, "b": 1e-06, "Ts": 0.05 # Зменшено Ts, щоб двійник рушав
    }

    @classmethod
    def ensure_dirs(cls):
        cls.AI_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. ЦИФРОВИЙ ДВІЙНИК (СИНХРОНІЗОВАНИЙ З ПРОШИВКОЮ)
# ============================================================================
class DigitalTwin:
    def __init__(self, config_path: Path = Config.CONFIG_FILE):
        self.config = self._load_config(config_path)
        m = self.config['motor_parameters']
        self.R, self.L, self.Ke = m.get('R', 4.5), m.get('L', 0.01), m.get('Ke', 0.6)
        self.Km, self.J, self.b, self.Ts = m.get('Km', 0.6), m.get('J', 0.001), m.get('b', 1e-6), m.get('Ts', 0.05)
        self.V_supply = self.config['hardware'].get('supply_voltage', 12.0)

    def _load_config(self, path: Path) -> Dict:
        if not path.exists():
            return {"hardware": {"supply_voltage": 12.0}, "motor_parameters": Config.DEFAULT_MOTOR_PARAMS}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def simulate_closed_loop(self, target: float, kp: float, ki: float, kd: float, x0: float = 0.0, dt: float = 0.01, T: float = 4.0) -> Tuple[float, float]:
        steps = int(T / dt)
        i, w = 0.0, x0
        e_sum, prev_e, d_filtered = 0.0, target - w, 0.0
        iae, max_w = 0.0, w
        current_pwm = 0.0
        max_internal_dt = 0.0005 # Підкроки 0.5 мс для стійкості
        
        for _ in range(steps):
            e = target - w
            # Інтегральне вікно (як у прошивці)
            if abs(e) < 6.0:
                e_sum = np.clip(e_sum + ki * e * dt, -2.0, 2.0)
            else:
                e_sum *= np.exp(-dt / 0.3)
            
            # D-фільтр
            de_raw = (e - prev_e) / dt
            prev_e = e
            d_filtered += (1.0 - np.exp(-dt / 0.02)) * (de_raw - d_filtered)
            
            i_ref = np.clip(e_sum + kp * e + kd * d_filtered, -2.0, 2.0)
            u = self.Ke * target + self.R * i_ref # Спрощений PBC для оптимізатора
            
            # Feedforward тертя
            if target > 0.1 and u > 0: u += 0.3 + 0.02 * abs(target)
            elif target < -0.1 and u < 0: u -= 0.3 + 0.02 * abs(target)
            
            target_pwm = np.clip(abs(u) / self.V_supply, 0.0, 1.0) * 255.0
            # Slew Rate
            max_delta = 1000.0 * dt
            if target_pwm > current_pwm + max_delta: current_pwm += max_delta
            elif target_pwm < current_pwm - max_delta: current_pwm -= max_delta
            else: current_pwm = target_pwm
            
            # Інтеграція з підкроками
            n_sub = max(1, int(np.ceil(dt / max_internal_dt)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                tau_e = self.L / max(self.R, 1e-9)
                i_next = ((u - self.Ke * w) / max(self.R, 1e-9)) + (i - ((u - self.Ke * w) / max(self.R, 1e-9))) * np.exp(-sub_dt / tau_e)
                torque = self.Km * i_next
                
                if abs(w) < 0.5 and abs(torque) < self.Ts:
                    dw = 0.0
                else:
                    friction = self.Ts * np.sign(torque if abs(w) < 1e-4 else w)
                    dw = (torque - friction - self.b * w) / max(self.J, 1e-12)
                
                i = i_next
                w += dw * sub_dt
                if np.isnan(w) or abs(w) > 200: w = 0.0
                if np.isnan(i) or abs(i) > 10: i = 0.0

            max_w = max(max_w, w)
            iae += abs(e) * dt
            
        overshoot = ((max_w - target) / target) * 100.0 if target > 0 else 0.0
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
                            writer.writerow([datetime.now().isoformat()] + list(data))
                            f.flush()
                        elif line: print(f"[Arduino]: {line}")

                threading.Thread(target=read_loop, daemon=True).start()

                if scenario_file:
                    print(f"🚀 Виконання сценарію: {scenario_file}")
                    with open(scenario_file, "r", encoding="utf-8") as f_scr:
                        for line in f_scr:
                            line = line.strip()
                            if not line or line.startswith("#"): continue
                            if line.lower().startswith("wait "):
                                time.sleep(float(line.split(" ")[1])); continue
                            if line.lower() == "stop_all": break
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
        # Шукаємо базові логи, виключаючи ті, що згенеровані ШІ
        all_logs = glob.glob(str(Config.LOGS_DIR / "*telemetry_*.csv"))
        csv_files = [f for f in all_logs if "ai_" not in Path(f).name and "hybrid_" not in Path(f).name]
        
        if not csv_files: 
            print("❌ Немає базових логів для калібрування."); return
        latest = max(csv_files, key=os.path.getctime)
        
        print(f"📋 Калібрування за файлом: {latest}...")
        df = pd.read_csv(latest)
        pwm, current, speed = df['PWM'].values, df['Current_A'].values, df['Omega'].values
        v_bus = df['Voltage'].values if 'Voltage' in df else np.full(len(df), 12.0)
        
        v_supply = round(float(v_bus[pwm == pwm.max()].mean()), 3)
        if v_supply < 1.0: v_supply = 12.0
        
        acceleration = np.abs(np.diff(speed, prepend=speed[0])) / 0.01
        steady = (pwm > 50) & (acceleration < 2.0)
        A = np.vstack([current[steady], speed[steady]]).T
        b_vec = v_supply * (pwm[steady] / 255.0)
        
        (R_cal, Ke_cal), _ = nnls(A, b_vec)
        R_cal, Ke_cal = max(R_cal, 1.0), max(Ke_cal, 0.01)
        
        # Зменшуємо Ts, щоб симуляція гарантовано рушала з місця
        params = {
            "hardware": {"supply_voltage": v_supply},
            "motor_parameters": {
                "R": round(R_cal, 3), "L": 0.01, "Ke": round(Ke_cal, 5), 
                "Km": round(Ke_cal, 5), "J": 0.001, "b": 1e-06, "Ts": 0.05 
            }
        }
        with open(Config.CONFIG_FILE, "w") as f: json.dump(params, f, indent=4)
        print(f"✅ Двійник відкалібровано! V_supply={v_supply}V, R={R_cal:.3f} Ом, Ke={Ke_cal:.4f}")

class AITrainer:
    @staticmethod
    def run(max_target: float = 15.0): # Дефолт 15.0, бо 21 недосяжне
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from sklearn.model_selection import train_test_split
        
        print(f"🧠 Генерація даних (від 3.0 до {max_target} рад/с) та навчання ШІ...")
        twin = DigitalTwin()
        bounds = [Config.BOUNDS_KP, Config.BOUNDS_KI, Config.BOUNDS_KD]
        X, Y = [], []
        
        targets = np.linspace(3.0, max_target, 13) 
        
        for i, target in enumerate(targets):
            print(f"  Оптимізація для {target:.1f} рад/с ({i+1}/{len(targets)})")
            def obj(g): 
                iae, over = twin.simulate_closed_loop(target, *g, x0=0.0)
                return iae + ((over - 15.0) * Config.PENALTY_OVERSHOOT if over > 15.0 else 0.0)
            
            # Фіксуємо seed для відтворюваності
            res = differential_evolution(obj, bounds, maxiter=10, popsize=8, seed=42+i)
            
            # Додаємо різні початкові швидкості (x0) для наукової новизни
            for _ in range(40):
                x0 = np.random.uniform(0, target * 0.5) 
                err = target - x0
                X.append([target, err])
                Y.append(res.x)
                
        X, Y = np.array(X), np.array(Y)
        X_m, X_s, Y_m, Y_s = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
        X_n = (X - X_m) / np.where(X_s == 0, 1, X_s)
        Y_n = (Y - Y_m) / np.where(Y_s == 0, 1, Y_s)
        
        # Валідаційна вибірка
        X_train, X_val, Y_train, Y_val = train_test_split(X_n, Y_n, test_size=0.2, random_state=42)
        
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
        import tensorflow as tf
        
        if not Config.MODEL_FILE.exists(): 
            print("❌ Модель не знайдена."); return
            
        model = tf.keras.models.load_model(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'rb') as f: scaler = pickle.load(f)
        
        mode_name = "Hybrid (Static AI)" if static_mode else "Dynamic AI"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Config.LOGS_DIR / f"{tag}_telemetry_{timestamp}.csv"
        
        try:
            with SerialManager(port) as hw, open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Current_A", "Voltage", "Omega", "Target", "PWM", "Kp", "Ki", "Kd"])
                
                print(f"🚀 HIL-тест ({mode_name}): Ціль: {target_speed} рад/с")
                hw.send("cfreq 100")
                
                # Якщо є сценарій - запускаємо його, інакше просто 'w'
                if scenario_file:
                    threading.Thread(target=lambda: [hw.send(l.strip()) or time.sleep(0.05) for l in open(scenario_file) if l.strip() and not l.startswith("#")], daemon=True).start()
                else:
                    hw.send(f"w {target_speed}")
                
                last_tune, kp, ki, kd = time.time(), 0.4, 0.1, 0.002
                is_calibrated = False
                
                while True:
                    line = hw.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) == 7: # Суворо 7 полів
                            curr, volt, omega, target, pwm = map(float, parts[:5])
                            
                            need_update = (not static_mode and time.time() - last_tune > 0.1) or (static_mode and not is_calibrated)
                            
                            if need_update and target != 0:
                                error = target - omega
                                X = np.array([[target, error]])
                                X_n = (X - scaler['X_mean']) / np.where(scaler['X_std'] == 0, 1, scaler['X_std'])
                                Y = model.predict(X_n, verbose=0) * scaler['Y_std'] + scaler['Y_mean']
                                
                                kp, ki, kd = np.clip(Y[0], [Config.BOUNDS_KP[0], Config.BOUNDS_KI[0], Config.BOUNDS_KD[0]], [Config.BOUNDS_KP[1], Config.BOUNDS_KI[1], Config.BOUNDS_KD[1]])
                                
                                hw.send(f"kp {kp:.3f}"); hw.send(f"ki {ki:.3f}"); hw.send(f"kd {kd:.5f}")
                                print(f"[{mode_name}] Target: {target:4.1f} | Error: {error:5.2f} | Kp={kp:.3f}, Ki={ki:.3f}")
                                last_tune = time.time()
                                is_calibrated = True
                            
                            writer.writerow([datetime.now().isoformat(), curr, volt, omega, target, pwm, f"{kp:.4f}", f"{ki:.4f}", f"{kd:.5f}"])
                            f.flush()
                            
        except KeyboardInterrupt:
            print(f"\n🛑 Зупинка...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Лог збережено у {csv_path}")
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
        
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
        print(f"✅ Графік збережено: {save_path}")

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
        
        plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
        print(f"✅ Порівняльний графік збережено: {save_path}")

class ReportGenerator:
    @staticmethod
    def create_multi_scenario_pdf(data_list: list, save_path: str = "logs/Final_Multi_Report.pdf"):
        import platform
        from fpdf import FPDF # Імпорт всередині, щоб не ламало collect
        
        pdf = FPDF()
        font_path = "C:/Windows/Fonts/arial.ttf" if platform.system() == "Windows" else "DejaVuSans.ttf"
        
        pdf.add_page()
        try:
            pdf.add_font("Arial", "", font_path, uni=True)
            pdf.set_font("Arial", size=20)
        except:
            print("⚠️ Шрифт не знайдено, використовуємо стандартний.")
            pdf.set_font("Helvetica", size=20)
            
        pdf.ln(60)
        pdf.cell(200, 20, txt="ЗВІТ ПРО РЕЗУЛЬТАТИ HIL-ДОСЛІДЖЕННЯ", ln=True, align='C')
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Порівняння Baseline, Hybrid та Dynamic AI", ln=True, align='C')

        def calc_metrics(df):
            t = (pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['Timestamp']).iloc[0]).dt.total_seconds().values
            error = df['Target'].values - df['Omega'].values
            iae = np.trapezoid(np.abs(error), x=t) # Сучасна функція
            # Максимальне перерегулювання
            overshoot = max(0, (df['Omega'].max() - df['Target'].max()) / (df['Target'].max() + 1e-9) * 100)
            # Статична похибка (останні 10% часу)
            tail = int(len(df) * 0.1)
            steady_err = np.mean(np.abs(df['Target'].values[-tail:] - df['Omega'].values[-tail:]))
            return iae, overshoot, steady_err

        for idx, data in enumerate(data_list):
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt=f"Сценарій: {data['name']}", ln=True)
            pdf.ln(5)

            df_b, df_h, df_a = pd.read_csv(data['baseline']), pd.read_csv(data['hybrid']), pd.read_csv(data['ai'])
            iae_b, os_b, ss_b = calc_metrics(df_b)
            iae_h, os_h, ss_h = calc_metrics(df_h)
            iae_a, os_a, ss_a = calc_metrics(df_a)

            pdf.set_font("Arial", size=10)
            pdf.cell(50, 10, txt="Метрика", border=1, align='C')
            pdf.cell(45, 10, txt="Baseline", border=1, align='C')
            pdf.cell(45, 10, txt="Hybrid", border=1, align='C')
            pdf.cell(45, 10, txt="Dynamic AI", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, txt="IAE (Інтеграл)", border=1)
            pdf.cell(45, 10, txt=f"{iae_b:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{iae_h:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{iae_a:.2f}", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, txt="Перерегулювання", border=1)
            pdf.cell(45, 10, txt=f"{os_b:.1f}%", border=1, align='C')
            pdf.cell(45, 10, txt=f"{os_h:.1f}%", border=1, align='C')
            pdf.cell(45, 10, txt=f"{os_a:.1f}%", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, txt="Стат. похибка", border=1)
            pdf.cell(45, 10, txt=f"{ss_b:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{ss_h:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{ss_a:.2f}", border=1, align='C')
            pdf.ln(15)

            # Тимчасовий графік у LOGS_DIR
            t_b = (pd.to_datetime(df_b['Timestamp']) - pd.to_datetime(df_b['Timestamp']).iloc[0]).dt.total_seconds().values
            t_h = (pd.to_datetime(df_h['Timestamp']) - pd.to_datetime(df_h['Timestamp']).iloc[0]).dt.total_seconds().values
            t_a = (pd.to_datetime(df_a['Timestamp']) - pd.to_datetime(df_a['Timestamp']).iloc[0]).dt.total_seconds().values

            plt.figure(figsize=(10, 4))
            plt.plot(t_b, df_b['Omega'], 'b-', alpha=0.5, label='Baseline')
            plt.plot(t_h, df_h['Omega'], 'orange', linewidth=1.5, label='Hybrid')
            plt.plot(t_a, df_a['Omega'], 'r-', linewidth=2, label='Dynamic AI')
            plt.plot(t_b, df_b['Target'], 'k--', alpha=0.5, label='Target')
            plt.title(f'Динаміка: {data["name"]}'); plt.ylabel('Швидкість'); plt.legend(); plt.grid(True)
            plt.tight_layout()
            
            chart_path = Config.LOGS_DIR / f"temp_chart_{idx}.png"
            plt.savefig(chart_path, dpi=150); plt.close()
            
            pdf.image(str(chart_path), x=15, w=180)
            os.remove(chart_path) # Видаляємо сміття

        pdf.output(save_path)
        print(f"✅ Звіт збережено: {save_path}")

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
    p_train.add_argument("--max-target", type=float, default=15.0, help="Фізична межа N20 - близько 18 рад/с")

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
            {"name": "Еталонний запуск", "baseline": get_latest("base_scenario"), "hybrid": get_latest("hybrid_scenario"), "ai": get_latest("ai_scenario")},
            {"name": "Стрес-тест", "baseline": get_latest("base_scenario-stress"), "hybrid": get_latest("hybrid_scenario-stress"), "ai": get_latest("ai_scenario-stress")}
        ]
        data_list = [d for d in data_list if all(d.values())]
        if data_list: ReportGenerator.create_multi_scenario_pdf(data_list, "logs/FINAL_THESIS_REPORT.pdf")
        else: print("❌ Не знайдено повних наборів даних.")