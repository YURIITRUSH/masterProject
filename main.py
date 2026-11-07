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
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional

import platform
from fpdf import FPDF
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Використовуємо фоновий режим для збереження графіків без блокування вікна
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
from scipy.optimize import differential_evolution, nnls

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# 0. ЦЕНТРАЛІЗОВАНА КОНФІГУРАЦІЯ (Всі налаштування в одному місці)
# ============================================================================
class Config:
    """
    Головний клас конфігурації проєкту.
    Змінюючи значення тут, ви змінюєте поведінку всієї системи.
    """
    # --- Шляхи до файлів ---
    BASE_DIR = Path(__file__).resolve().parent
    AI_DIR = BASE_DIR / "ai"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_FILE = AI_DIR / "config.json"
    MODEL_FILE = AI_DIR / "tf_model.keras"
    SCALER_FILE = AI_DIR / "tf_scaler.pkl"

    # --- Параметри Serial-порту ---
    BAUD_RATE = 115200
    TELEMETRY_REGEX = re.compile(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?){3,}$')

    # --- Обмеження для пошуку коефіцієнтів PID (Differential Evolution) ---
    # Формат: (мінімум, максимум)
    BOUNDS_KP = (0.1, 1.5)
    BOUNDS_KI = (0.01, 1.0)
    BOUNDS_KD = (0.0001, 0.02)
    PENALTY_OVERSHOOT = 500.0 # Штраф за перерегулювання більше 15%

    # --- Фізичні параметри за замовчуванням (якщо не вдалося відкалібрувати) ---
    DEFAULT_MOTOR_PARAMS = {
        "R": 4.5, "L": 0.01, "Ke": 0.6, "Km": 0.6, 
        "J": 0.001, "b": 1e-06, "Ts": 0.15
    }

    @classmethod
    def ensure_dirs(cls):
        """Створює необхідні папки, якщо їх немає."""
        cls.AI_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. МАТЕМАТИЧНА МОДЕЛЬ (DIGITAL TWIN)
# ============================================================================
class DigitalTwin:
    """
    Цифровий двійник DC-двигуна постійного струму.
    Використовує систему диференціальних рівнянь для симуляції.
    """
    def __init__(self, config_path: Path = Config.CONFIG_FILE):
        self.config = self._load_config(config_path)
        m = self.config['motor_parameters']
        
        # Фізичні параметри
        self.R = m.get('R', 4.5)
        self.L = m.get('L', 0.01)
        self.Ke = m.get('Ke', 0.6)
        self.Km = m.get('Km', 0.6)
        self.J = m.get('J', 0.001)
        self.b = m.get('b', 1e-6)
        self.Ts = m.get('Ts', 0.15)
        self.V_supply = self.config['hardware'].get('supply_voltage', 12.0)

    def _load_config(self, path: Path) -> Dict:
        if not path.exists():
            print(f"⚠️ Конфіг не знайдено. Використовуємо дефолтні параметри.")
            return {
                "hardware": {"supply_voltage": 12.0},
                "motor_parameters": Config.DEFAULT_MOTOR_PARAMS
            }
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _physics_step(self, u: float, i: float, omega: float, dt: float) -> Tuple[float, float]:
        """
        Один крок інтеграції фізичної моделі.
        Використовує аналітичне рішення для струму (через жорсткість системи) 
        та метод Ейлера для механіки.
        """
        # 1. Електрична частина: L*di/dt = u - R*i - Ke*omega
        tau_e = self.L / max(self.R, 1e-9)
        u_eq = u - self.Ke * omega
        i_inf = u_eq / max(self.R, 1e-9)
        i_next = i_inf + (i - i_inf) * np.exp(-dt / max(tau_e, 1e-9))
        
        # 2. Механічна частина: J*dw/dt = Km*i - Ts*sign(w) - b*w
        torque = self.Km * i_next
        if abs(omega) < 0.5 and abs(torque) < self.Ts:
            dw_dt = 0.0 # Мотор стоїть (сухе тертя тримає його)
        else:
            friction = self.Ts * np.sign(torque if abs(omega) < 1e-4 else omega)
            dw_dt = (torque - friction - self.b * omega) / max(self.J, 1e-12)
            
        omega_next = omega + dw_dt * dt

        # Захист від чисельного вибуху
        if np.isnan(omega_next) or abs(omega_next) > 200: omega_next = 0.0
        if np.isnan(i_next) or abs(i_next) > 10: i_next = 0.0
        return i_next, omega_next

    def simulate_closed_loop(self, target: float, kp: float, ki: float, kd: float, dt: float = 0.01, T: float = 4.0) -> Tuple[float, float]:
        """
        Симуляція замкнутого контуру PID-PBC (використовується оптимізатором для пошуку Kp/Ki/Kd).
        Повертає: (IAE - інтеграл абсолютної похибки, Overshoot - перерегулювання у %).
        """
        steps = int(T / dt)
        i, w = 0.0, 0.0
        e_sum, prev_e = 0.0, target - w
        iae, max_w = 0.0, 0.0
        
        for _ in range(steps):
            e = target - w
            e_sum = np.clip(e_sum + ki * e * dt, -2.0, 2.0) # Anti-windup
            D = kd * (e - prev_e) / dt
            prev_e = e
            
            # Закон керування (спрощений PBC)
            i_ref = e_sum + kp * e + D
            u = np.clip(self.Ke * target + self.R * i_ref, -self.V_supply, self.V_supply)
            
            i, w = self._physics_step(u, i, w, dt)
            max_w = max(max_w, w)
            iae += abs(e) * dt
            
        overshoot = ((max_w - target) / target) * 100.0 if target > 0 else 0.0
        return iae, max(overshoot, 0.0)

# ============================================================================
# 2. КОНТЕКСТНИЙ МЕНЕДЖЕР ДЛЯ SERIAL (Безпечна робота з портом)
# ============================================================================
class SerialManager:
    """
    Клас для безпечного підключення до Arduino.
    Гарантує закриття порту та зупинку мотора навіть при помилках (через __exit__).
    """
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
            raise ConnectionError("❌ Arduino не знайдено. Вкажіть порт вручну (наприклад, --port COM5).")
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2) # Чекаємо на перезавантаження Arduino
            self.ser.reset_input_buffer()
            print(f"🔌 Успішно підключено до {self.port}")
            return self
        except serial.SerialException as e:
            raise ConnectionError(f"❌ Помилка відкриття {self.port}: {e}\n💡 Перевірте Serial Monitor!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self.ser and self.ser.is_open:
            try: 
                self.ser.write(b"s\n") # Безпечна зупинка
                time.sleep(0.1)
            except: pass
            self.ser.close()
            print("🔒 COM-порт закрито.")

    def send(self, cmd: str):
        if self.ser and self.ser.is_open:
            self.ser.write(f"{cmd}\n".encode())

# ============================================================================
# 3. МОДУЛІ СИСТЕМИ (Кожен відповідає за свій етап)
# ============================================================================
class DataCollector:
    """Етап 1: Збір телеметрії з реального заліза."""
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
                        elif line: 
                            print(f"[Arduino]: {line}")

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
                            hw.send(line)
                            time.sleep(0.05)
                else:
                    print("💡 Ручний режим. Введіть 'w 8.0', 's', 'status' (Ctrl+C для виходу).")
                    while True: hw.send(input())
                    
        except KeyboardInterrupt:
            print("\n🛑 Зупинка...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Дані збережено у {csv_path}")
            if auto_plot: Analyzer.run(file_path=str(csv_path))

class Calibrator:
    """Етап 2: Ідентифікація параметрів Digital Twin."""
    @staticmethod
    def run():
        csv_files = glob.glob(str(Config.LOGS_DIR / "telemetry_*.csv"))
        if not csv_files: 
            print("❌ Немає базових логів для калібрування."); return
        latest = max(csv_files, key=os.path.getctime)
        
        print(f"📋 Калібрування за файлом: {latest}...")
        df = pd.read_csv(latest)
        pwm, current, speed = df['PWM'].values, df['Current_A'].values, df['Omega'].values
        v_bus = df['Voltage'].values if 'Voltage' in df else np.full(len(df), 12.0)
        
        # 1. Ефективна напруга
        v_supply = round(float(v_bus[pwm == pwm.max()].mean()), 3)
        if v_supply < 1.0: v_supply = 12.0
        
        # 2. Електричні параметри (Non-Negative Least Squares)
        acceleration = np.abs(np.diff(speed, prepend=speed[0])) / 0.01
        steady = (pwm > 50) & (acceleration < 2.0)
        A = np.vstack([current[steady], speed[steady]]).T
        b_vec = v_supply * (pwm[steady] / 255.0)
        
        (R_cal, Ke_cal), _ = nnls(A, b_vec)
        R_cal, Ke_cal = max(R_cal, 1.0), max(Ke_cal, 0.01)
        
        params = {
            "hardware": {"port": "auto", "baudrate": Config.BAUD_RATE, "supply_voltage": v_supply},
            "motor_parameters": {
                "R": round(R_cal, 3), "L": 0.01, "Ke": round(Ke_cal, 5), 
                "Km": round(Ke_cal, 5), "J": 0.001, "b": 1e-06, "Ts": 0.15
            }
        }
        with open(Config.CONFIG_FILE, "w") as f: json.dump(params, f, indent=4)
        print(f"✅ Двійник відкалібровано! V_supply={v_supply}V, R={R_cal:.3f} Ом, Ke={Ke_cal:.4f}")

class AITrainer:
    """Етап 3: Генерація даних на двійнику та навчання нейромережі."""
    @staticmethod
    def run(max_target: float = 21.0):
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        print(f"🧠 Генерація даних (від 3.0 до {max_target} рад/с) та навчання ШІ...")
        twin = DigitalTwin()
        bounds = [Config.BOUNDS_KP, Config.BOUNDS_KI, Config.BOUNDS_KD]
        X, Y = [], []
        
        num_points = 19 # Кількість точок для генерації
        targets = np.linspace(3.0, max_target, num_points) 
        
        for i, target in enumerate(targets):
            print(f"  Оптимізація для {target:.1f} рад/с ({i+1}/{len(targets)})")
            def obj(g): 
                iae, over = twin.simulate_closed_loop(target, *g)
                return iae + ((over - 15.0) * Config.PENALTY_OVERSHOOT if over > 15.0 else 0.0)
            
            res = differential_evolution(obj, bounds, maxiter=10, popsize=8)
            for _ in range(50): # Додаємо шум для різноманітності
                X.append([target, np.random.uniform(-2.0, 2.0)])
                Y.append(res.x)
                
        X, Y = np.array(X), np.array(Y)
        X_m, X_s, Y_m, Y_s = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
        X_n = (X - X_m) / np.where(X_s == 0, 1, X_s)
        Y_n = (Y - Y_m) / np.where(Y_s == 0, 1, Y_s)
        
        model = models.Sequential([
            layers.Input((2,)), 
            layers.Dense(32, activation='relu'), 
            layers.Dense(16, activation='relu'), 
            layers.Dense(3)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        history = model.fit(X_n, Y_n, epochs=50, batch_size=16, verbose=1)
        
        model.save(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'wb') as f:
            pickle.dump({'X_mean': X_m, 'X_std': X_s, 'Y_mean': Y_m, 'Y_std': Y_s}, f)
        
        # Автоматична побудова графіка навчання
        plt.figure(figsize=(8, 5))
        plt.plot(history.history['loss'], label='Train Loss', color='purple')
        plt.title('Історія навчання нейромережі')
        plt.xlabel('Епоха'); plt.ylabel('Помилка (MSE)')
        plt.legend(); plt.grid(True)
        plt.savefig(Config.LOGS_DIR / "training_history.png", dpi=150)
        plt.close()
        print(f"✅ Модель збережена! Графік навчання: {Config.LOGS_DIR / 'training_history.png'}")

class HILController:
    @staticmethod
    def run(target_speed: float, port: Optional[str] = None, auto_plot: bool = False, static_mode: bool = False, tag: str = "ai"):
        import tensorflow as tf
        
        if not Config.MODEL_FILE.exists(): 
            print("❌ Модель не знайдена. Спочатку запустіть train."); return
            
        model = tf.keras.models.load_model(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'rb') as f: 
            scaler = pickle.load(f)
        
        mode_name = "Hybrid (Static AI)" if static_mode else "Dynamic AI"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Використовуємо tag напряму, щоб уникнути дублювання префіксів
        csv_path = Config.LOGS_DIR / f"{tag}_telemetry_{timestamp}.csv"
        
        try:
            with SerialManager(port) as hw, open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Current_A", "Voltage", "Omega", "Target", "PWM", "Kp", "Ki", "Kd"])
                
                print(f"🚀 HIL-тест ({mode_name}): Ціль: {target_speed} рад/с")
                hw.send("cfreq 100")
                hw.send(f"w {target_speed}")
                
                last_tune, kp, ki, kd = time.time(), 0.4, 0.1, 0.002
                is_calibrated = False
                
                while True:
                    line = hw.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 7:
                            curr, volt, omega, target, pwm = map(float, parts[:5])
                            
                            # Логіка оновлення (для static_mode оновлюємо лише 1 раз)
                            need_update = (not static_mode and time.time() - last_tune > 0.1) or (static_mode and not is_calibrated)
                            
                            if need_update and target != 0:
                                error = target - omega
                                X = np.array([[target, error]])
                                X_n = (X - scaler['X_mean']) / np.where(scaler['X_std'] == 0, 1, scaler['X_std'])
                                Y = model.predict(X_n, verbose=0) * scaler['Y_std'] + scaler['Y_mean']
                                
                                kp, ki, kd = np.clip(Y[0], 
                                                     [Config.BOUNDS_KP[0], Config.BOUNDS_KI[0], Config.BOUNDS_KD[0]], 
                                                     [Config.BOUNDS_KP[1], Config.BOUNDS_KI[1], Config.BOUNDS_KD[1]])
                                
                                hw.send(f"kp {kp:.3f}"); hw.send(f"ki {ki:.3f}"); hw.send(f"kd {kd:.5f}")
                                print(f"[{mode_name}] Target: {target:4.1f} | Error: {error:5.2f} | Kp={kp:.3f}, Ki={ki:.3f}")
                                last_tune = time.time()
                                is_calibrated = True
                            
                            writer.writerow([datetime.now().isoformat(), curr, volt, omega, target, pwm, f"{kp:.4f}", f"{ki:.4f}", f"{kd:.5f}"])
                            f.flush()
                            
        except KeyboardInterrupt:
            print(f"\n🛑 Зупинка {mode_name}...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Лог {mode_name} збережено у {csv_path}")
            if auto_plot: Analyzer.run(file_path=str(csv_path))

class Analyzer:
    """Етап 5: Візуалізація результатів (універсальна для всіх етапів)."""
    @staticmethod
    def run(file_path: Optional[str] = None, save_path: str = "logs/final_ai_result.png"):
        if not file_path:
            files = glob.glob(str(Config.LOGS_DIR / "*telemetry*.csv"))
            if not files: print("❌ Немає CSV-логів для аналізу."); return
            file_path = max(files, key=os.path.getctime)
            
        print(f"📊 Побудова графіка для {file_path}...")
        df = pd.read_csv(file_path)
        
        # Конвертація часу
        t = (pd.to_datetime(df['Timestamp']) - pd.to_datetime(df['Timestamp']).iloc[0]).dt.total_seconds().values
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        has_gains = 'Kp' in df.columns
        
        if has_gains:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 1. Швидкість
        ax1.plot(t, df['Omega'], 'b-', linewidth=2, label='Фактична швидкість')
        ax1.plot(t, df['Target'], 'r--', linewidth=1.5, label='Цільова швидкість')
        ax1.set_ylabel('Швидкість (рад/с)', fontsize=11)
        ax1.legend(loc='lower right'); ax1.grid(True, linestyle=':')
        
        # 2. ШІМ
        ax2.plot(t, df['PWM'], 'g-', linewidth=1.5, label='PWM (0-255)')
        ax2.set_ylabel('ШІМ', fontsize=11)
        ax2.legend(loc='lower right'); ax2.grid(True, linestyle=':')
        
        # 3. Струм
        ax3.plot(t, df['Current_A'], 'm-', linewidth=1.5, label='Струм (А)')
        ax3.set_ylabel('Струм (А)', fontsize=11)
        ax3.grid(True, linestyle=':')
        if not has_gains:
            ax3.set_xlabel('Час (с)', fontsize=11)
            ax3.legend(loc='upper right')
        
        # 4. Коефіцієнти PID (тільки якщо є дані)
        if has_gains:
            ax4.plot(t, df['Kp'], 'c-', linewidth=2, label='Kp')
            ax4.plot(t, df['Ki'], 'm-', linewidth=2, label='Ki')
            # Множимо Kd на 100 для наочності, щоб він був на одному масштабі з іншими
            ax4.plot(t, df['Kd'] * 100, 'y--', linewidth=2, label='Kd × 100') 
            
            ax4.set_ylabel('Коефіцієнти PID', fontsize=11)
            ax4.set_xlabel('Час (с)', fontsize=11)
            ax4.legend(loc='upper right')
            ax4.grid(True, linestyle=':')
        
        plt.suptitle(f"Аналіз: {Path(file_path).name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Графік успішно збережено: {save_path}")
    
    @staticmethod
    def compare(file1: str, file2: str, save_path: str = "logs/comparison.png"):
        """Накладання графіків двох файлів на одне полотно."""
        print(f"📊 Порівняння: {Path(file1).name} vs {Path(file2).name}...")
        
        df_b = pd.read_csv(file1)
        df_a = pd.read_csv(file2)
        
        # Конвертація часу в секунди від початку
        t_b = (pd.to_datetime(df_b['Timestamp']) - pd.to_datetime(df_b['Timestamp']).iloc[0]).dt.total_seconds().values
        t_a = (pd.to_datetime(df_a['Timestamp']) - pd.to_datetime(df_a['Timestamp']).iloc[0]).dt.total_seconds().values
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # --- Верхній графік: Швидкість ---
        ax1.plot(t_b, df_b['Omega'], 'b-', linewidth=2, alpha=0.7, label='Baseline (Ручний PID)')
        ax1.plot(t_a, df_a['Omega'], 'r-', linewidth=2, label='Neural Gain Scheduling (ШІ)')
        
        # Малюємо цільову лінію (беремо з Baseline, бо вона однакова)
        ax1.plot(t_b, df_b['Target'], 'k--', linewidth=1.5, alpha=0.6, label='Цільова швидкість')
        
        ax1.set_ylabel('Швидкість (рад/с)', fontsize=12)
        ax1.legend(loc='lower right', fontsize=11)
        ax1.grid(True, linestyle=':')
        ax1.set_title('Порівняння перехідних процесів: Ручний PID vs Нейромережа', fontsize=14, fontweight='bold')
        
        # --- Нижній графік: ШІМ (Енергоспоживання) ---
        ax2.plot(t_b, df_b['PWM'], 'b-', linewidth=1.5, alpha=0.7, label='Baseline (Ручний PID)')
        ax2.plot(t_a, df_a['PWM'], 'g-', linewidth=2, label='Neural Gain Scheduling (ШІ)')
        ax2.set_ylabel('ШІМ (0-255)', fontsize=12)
        ax2.set_xlabel('Час (с)', fontsize=12)
        ax2.legend(loc='lower right', fontsize=11)
        ax2.grid(True, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ Порівняльний графік збережено: {save_path}")

class ReportGenerator:
    @staticmethod
    def create_multi_scenario_pdf(data_list: list, save_path: str = "logs/Final_Multi_Report.pdf"):
        print(f"📄 Генерація багатосторінкового PDF-звіту...")
        import matplotlib.pyplot as plt

        pdf = FPDF()
        font_path = "C:/Windows/Fonts/arial.ttf" if platform.system() == "Windows" else "DejaVuSans.ttf"
        
        # Титульна сторінка
        pdf.add_page()
        try:
            pdf.add_font("Arial", "", font_path, uni=True)
            pdf.set_font("Arial", size=20)
        except:
            pdf.set_font("Helvetica", size=20)
            
        pdf.ln(60)
        pdf.cell(200, 20, txt="ЗВІТ ПРО РЕЗУЛЬТАТИ HIL-ДОСЛІДЖЕННЯ", ln=True, align='C')
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Порівняння Baseline, Hybrid та Dynamic AI", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Дата: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')

        for idx, data in enumerate(data_list):
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt=f"Сценарій {idx+1}: {data['name']}", ln=True)
            pdf.ln(5)

            df_b = pd.read_csv(data['baseline'])
            df_h = pd.read_csv(data['hybrid'])
            df_a = pd.read_csv(data['ai'])

            def calc_metrics(df):
                target = df['Target'].max() 
                if target == 0: target = 15.0
                error = target - df['Omega'].values
                iae = np.trapz(np.abs(error), dx=0.01)
                peak = df['Omega'].max()
                overshoot = max(0, (peak - target) / target * 100)
                return iae, overshoot

            iae_b, os_b = calc_metrics(df_b)
            iae_h, os_h = calc_metrics(df_h)
            iae_a, os_a = calc_metrics(df_a)

            pdf.set_font("Arial", size=10)
            pdf.cell(50, 10, txt="Метрика", border=1, align='C')
            pdf.cell(45, 10, txt="Baseline (PID)", border=1, align='C')
            pdf.cell(45, 10, txt="Hybrid (Static AI)", border=1, align='C')
            pdf.cell(45, 10, txt="Dynamic AI", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, txt="IAE (Інтеграл похибки)", border=1)
            pdf.cell(45, 10, txt=f"{iae_b:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{iae_h:.2f}", border=1, align='C')
            pdf.cell(45, 10, txt=f"{iae_a:.2f}", border=1, align='C')
            pdf.ln()
            
            pdf.cell(50, 10, txt="Макс. перерегулювання", border=1)
            pdf.cell(45, 10, txt=f"{os_b:.1f}%", border=1, align='C')
            pdf.cell(45, 10, txt=f"{os_h:.1f}%", border=1, align='C')
            pdf.cell(45, 10, txt=f"{os_a:.1f}%", border=1, align='C')
            pdf.ln(15)

            t_b = (pd.to_datetime(df_b['Timestamp']) - pd.to_datetime(df_b['Timestamp']).iloc[0]).dt.total_seconds().values
            t_h = (pd.to_datetime(df_h['Timestamp']) - pd.to_datetime(df_h['Timestamp']).iloc[0]).dt.total_seconds().values
            t_a = (pd.to_datetime(df_a['Timestamp']) - pd.to_datetime(df_a['Timestamp']).iloc[0]).dt.total_seconds().values

            plt.figure(figsize=(10, 4))
            plt.plot(t_b, df_b['Omega'], 'b-', alpha=0.5, label='Baseline (PID)')
            plt.plot(t_h, df_h['Omega'], 'orange', linewidth=1.5, label='Hybrid (Static AI)')
            plt.plot(t_a, df_a['Omega'], 'r-', linewidth=2, label='Dynamic AI')
            plt.plot(t_b, df_b['Target'], 'k--', alpha=0.5, label='Target')
            plt.title(f'Динаміка: {data["name"]}')
            plt.ylabel('Швидкість (рад/с)'); plt.xlabel('Час (с)')
            plt.legend(); plt.grid(True)
            plt.tight_layout()
            
            chart_path = f"logs/temp_chart_{idx}.png"
            plt.savefig(chart_path, dpi=150)
            plt.close()
            
            pdf.image(chart_path, x=15, w=180)

        pdf.output(save_path)
        print(f"✅ Багатосторінковий звіт збережено: {save_path}")

# ============================================================================
# 4. ГОЛОВНИЙ КОНТРОЛЕР (CLI - Інтерфейс командного рядка)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Універсальний HIL-комплекс (Digital Twin + TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subs = parser.add_subparsers(dest="cmd", required=True, help="Доступні команди")

    # Команда collect
    p_col = subs.add_parser("collect", help="Збір телеметрії з реального мотора")
    p_col.add_argument("--port", default=None, help="COM-порт (наприклад, COM5)")
    p_col.add_argument("--scenario", default=None, help="Шлях до файлу сценарію (.txt)")
    p_col.add_argument("--plot", action="store_true", help="Автоматично побудувати графік після завершення")
    p_col.add_argument("--tag", default="base", help="Тег для назви файлу (наприклад, scenario1)")

    # Команда calibrate
    subs.add_parser("calibrate", help="Калібрування Digital Twin за останнім логом")

    # Команда train
    p_train = subs.add_parser("train", help="Генерація даних та навчання нейромережі")
    p_train.add_argument("--max-target", type=float, default=21.0, help="Максимальна цільова швидкість для навчання")

    # Команда ai
    p_ai = subs.add_parser("ai", help="HIL-тест (нейромережа керує мотором)")
    p_ai.add_argument("--port", default=None, help="COM-порт")
    p_ai.add_argument("--target", type=float, default=8.0, help="Цільова швидкість (рад/с)")
    p_ai.add_argument("--plot", action="store_true", help="Автоматично побудувати графік після завершення")
    p_ai.add_argument("--tag", default="ai", help="Тег для назви файлу")
    p_ai.add_argument("--static", action="store_true", help="Використовувати статичні (гібридні) коефіцієнти")
    
    # Команда analyze
    p_ana = subs.add_parser("analyze", help="Побудова графіків за існуючим логом")
    p_ana.add_argument("--file", default=None, help="Шлях до CSV-файлу")
    p_ana.add_argument("--save", default="logs/final_ai_result.png", help="Шлях для збереження PNG")
    
    # Команда compare
    p_cmp = subs.add_parser("compare", help="Накладання двох графіків для порівняння")
    p_cmp.add_argument("--file1", required=True, help="Шлях до CSV 1")
    p_cmp.add_argument("--file2", required=True, help="Шлях до CSV 2")
    p_cmp.add_argument("--save", default="logs/comparison.png", help="Шлях для збереження")

    # Команда final-report
    subs.add_parser("final-report", help="Генерація повного PDF звіту для всіх сценаріїв")
    
    args = parser.parse_args()

    # Маршрутизація команд
    if args.cmd == "collect": 
        DataCollector.run(args.scenario, args.port, args.plot, args.tag)
    elif args.cmd == "calibrate": 
        Calibrator.run()
    elif args.cmd == "train": 
        AITrainer.run(args.max_target)
    elif args.cmd == "ai": 
        HILController.run(args.target, args.port, args.plot, args.static, args.tag)
    elif args.cmd == "analyze":
        Analyzer.run(args.file, args.save)
    elif args.cmd == "compare":
        Analyzer.compare(args.file1, args.file2, args.save)
    elif args.cmd == "final-report":
        import glob
        # Знаходимо найсвіжіші файли для кожного типу
        def get_latest(prefix):
            files = glob.glob(str(Config.LOGS_DIR / f"{prefix}_*.csv"))
            return max(files, key=os.path.getctime) if files else None

        data_list = [
            {
                "name": "Еталонний запуск (scenario.txt)",
                "baseline": get_latest("base_scenario_telemetry"),
                "hybrid": get_latest("hybrid_scenario_telemetry"),
                "ai": get_latest("ai_scenario_telemetry")
            },
            {
                "name": "Стрес-тест (scenario-stress.txt)",
                "baseline": get_latest("base_scenario-stress_telemetry"),
                "hybrid": get_latest("hybrid_scenario-stress_telemetry"),
                "ai": get_latest("ai_scenario-stress_telemetry")
            }
        ]
        
        # Фільтруємо порожні, якщо якийсь тест не запускався
        data_list = [d for d in data_list if d["baseline"] and d["hybrid"] and d["ai"]]
        
        if data_list:
            ReportGenerator.create_multi_scenario_pdf(data_list, "logs/FINAL_THESIS_REPORT.pdf")
        else:
            print("❌ Не знайдено повних наборів даних (baseline, hybrid, ai). Перевірте логи.")