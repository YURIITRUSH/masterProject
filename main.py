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
    def run(scenario_file: Optional[str] = None, port: Optional[str] = None, auto_plot: bool = False):
        Config.ensure_dirs()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Config.LOGS_DIR / f"telemetry_{timestamp}.csv"
        
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
        print(f"✅ Модель збережена! Графік навчання: {Config.LOGS_DIR / 'training_history.png'}")

class HILController:
    """Етап 4: HIL-тест (нейромережа керує реальним мотором)."""
    @staticmethod
    def run(target_speed: float, port: Optional[str] = None, auto_plot: bool = False):
        import tensorflow as tf
        
        if not Config.MODEL_FILE.exists(): 
            print("❌ Модель не знайдена. Спочатку запустіть train."); return
            
        model = tf.keras.models.load_model(Config.MODEL_FILE)
        with open(Config.SCALER_FILE, 'rb') as f: 
            scaler = pickle.load(f)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Config.LOGS_DIR / f"ai_telemetry_{timestamp}.csv"
        
        try:
            with SerialManager(port) as hw, open(csv_path, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Current_A", "Voltage", "Omega", "Target", "PWM", "Kp", "Ki", "Kd"])
                
                print(f"🚀 HIL-тест: ШІ керує мотором. Ціль: {target_speed} рад/с")
                hw.send("cfreq 100")
                hw.send(f"w {target_speed}")
                
                last_tune, kp, ki, kd = time.time(), 0.4, 0.1, 0.002
                while True:
                    line = hw.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line and ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 7:
                            curr, volt, omega, target, pwm = map(float, parts[:5])
                            
                            # Оновлення коефіцієнтів раз на 100 мс
                            if time.time() - last_tune > 0.1 and target != 0:
                                error = target - omega
                                X = np.array([[target, error]])
                                X_n = (X - scaler['X_mean']) / np.where(scaler['X_std'] == 0, 1, scaler['X_std'])
                                Y = model.predict(X_n, verbose=0) * scaler['Y_std'] + scaler['Y_mean']
                                
                                kp, ki, kd = np.clip(Y[0], 
                                                     [Config.BOUNDS_KP[0], Config.BOUNDS_KI[0], Config.BOUNDS_KD[0]], 
                                                     [Config.BOUNDS_KP[1], Config.BOUNDS_KI[1], Config.BOUNDS_KD[1]])
                                
                                hw.send(f"kp {kp:.3f}"); hw.send(f"ki {ki:.3f}"); hw.send(f"kd {kd:.5f}")
                                print(f"Target: {target:4.1f} | Error: {error:5.2f} | Kp={kp:.3f}, Ki={ki:.3f}")
                                last_tune = time.time()
                            
                            writer.writerow([datetime.now().isoformat(), curr, volt, omega, target, pwm, f"{kp:.4f}", f"{ki:.4f}", f"{kd:.5f}"])
                            f.flush()
                            
        except KeyboardInterrupt:
            print("\n🛑 Зупинка HIL-тесту...")
        except Exception as e:
            print(f"\n❌ Помилка: {e}")
        finally:
            print(f"💾 Лог HIL-тесту збережено у {csv_path}")
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
        
        # 3. Струм та Kp (якщо є)
        ax3.plot(t, df['Current_A'], 'm-', linewidth=1.5, label='Струм (А)')
        ax3.set_ylabel('Струм (А)', fontsize=11)
        ax3.set_xlabel('Час (с)', fontsize=11)
        ax3.grid(True, linestyle=':')
        
        if 'Kp' in df.columns:
            ax4 = ax3.twinx()
            ax4.plot(t, df['Kp'], 'c--', alpha=0.6, label='Коефіцієнт Kp')
            ax4.set_ylabel('Kp (від нейромережі)', fontsize=11, color='c')
            ax4.tick_params(axis='y', labelcolor='c')
            ax4.legend(loc='upper right')
        else:
            ax3.legend(loc='upper right')
        
        plt.suptitle(f"Аналіз: {Path(file_path).name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ Графік успішно збережено: {save_path}")

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

    # Команда analyze
    p_ana = subs.add_parser("analyze", help="Побудова графіків за існуючим логом")
    p_ana.add_argument("--file", default=None, help="Шлях до CSV-файлу")
    p_ana.add_argument("--save", default="logs/final_ai_result.png", help="Шлях для збереження PNG")

    args = parser.parse_args()

    # Маршрутизація команд
    if args.cmd == "collect": 
        DataCollector.run(args.scenario, args.port, args.plot)
    elif args.cmd == "calibrate": 
        Calibrator.run()
    elif args.cmd == "train": 
        AITrainer.run(args.max_target)
    elif args.cmd == "ai": 
        HILController.run(args.target, args.port, args.plot)
    elif args.cmd == "analyze": 
        Analyzer.run(args.file, args.save)