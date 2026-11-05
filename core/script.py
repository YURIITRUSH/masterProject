import csv
import logging
import logging.handlers
import os
import queue
import re
import sys
import threading
import time
import argparse
from collections import deque
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ============================================================================
# КОНФІГУРАЦІЯ
# ============================================================================
class Config:
    BAUD_RATE = 115200
    TIMEOUT = 0.1
    MAX_POINTS = 300
    LOG_DIR = "logs"
    LOG_WITH_TIMESTAMP = True

    ENABLE_PLOT = False
    FLUSH_EVERY_N_ROWS = 10

# 4 або більше чисел через кому — новий формат: Current,Omega,Target,PWM,Ticks,dTicks
TELEMETRY_RE = re.compile(r'^-?\d+(\.\d+)?(,-?\d+(\.\d+)?){3,}$')

# ============================================================================
# КЛАС КЕРУВАННЯ SERIAL-ПОРТОМ
# ============================================================================
class SerialManager:
    def __init__(self, baud_rate, port=None):
        self.baud_rate = baud_rate
        self.port = port
        self.ser = None
        self._partial = b""

    def connect(self):
        port = self.port or self._find_port()
        if not port:
            logging.error("COM-порт не знайдено.")
            return False
        try:
            self.ser = serial.Serial(port, self.baud_rate, timeout=Config.TIMEOUT)
            time.sleep(2)
            self.ser.reset_input_buffer()
            logging.info(f"Успішно підключено до {port}")
            return True
        except serial.SerialException as e:
            logging.error(f"Помилка підключення: {e}")
            return False

    def _find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            desc = (port.description or "").lower()
            if any(k in desc for k in ["arduino", "ch340", "cp210", "usb"]):
                return port.device
        return ports[0].device if ports else None

    def read_loop(self, stop_event, on_sample):
        while not stop_event.is_set():
            try:
                if self.ser.in_waiting == 0:
                    time.sleep(0.001)
                    continue
                raw = self._partial + self.ser.readline()
            except (serial.SerialException, OSError) as e:
                logging.error(f"Втрачено зв'язок із портом: {e}")
                stop_event.set()
                break

            if not raw.endswith(b'\n'):
                self._partial = raw
                continue
            self._partial = b""

            line = raw.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            if TELEMETRY_RE.match(line):
                data = tuple(float(p) for p in line.split(','))
                on_sample(time.time(), data)
            else:
                print(f"[Arduino]: {line}")

    def send(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(f"{cmd}\n".encode('utf-8'))
            logging.info(f"Відправлено: {cmd}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            logging.info("COM-порт закрито.")

# ============================================================================
# КЛАС ЛОГУВАННЯ
# ============================================================================
class DataLogger:
    def __init__(self, prefix, with_timestamp=False):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_file = os.path.join(Config.LOG_DIR, f"{prefix}_{timestamp}.csv")
        self.with_timestamp = with_timestamp

        self._lock = threading.Lock()
        self._closed = False
        self.rows = 0
        self._rows_since_flush = 0
        self._first_ts = None
        self._last_ts = None

        self.file_handle = open(self.telemetry_file, mode='w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.file_handle)
        header = ["Current_A", "Omega", "Target", "PWM", "Ticks", "dTicks"]
        if with_timestamp:
            header.insert(0, "Timestamp")
        self.csv_writer.writerow(header)
        self.file_handle.flush()

    def log(self, data, ts):
        row = list(data)
        if self.with_timestamp:
            row.insert(0, datetime.fromtimestamp(ts).isoformat())

        with self._lock:
            if self._closed: return
            self.csv_writer.writerow(row)
            self.rows += 1
            self._rows_since_flush += 1

            if self._first_ts is None: self._first_ts = ts
            self._last_ts = ts

            if self._rows_since_flush >= Config.FLUSH_EVERY_N_ROWS:
                self.file_handle.flush()
                self._rows_since_flush = 0

    def close(self):
        with self._lock:
            if self._closed: return
            self._closed = True
            self.file_handle.flush()
            self.file_handle.close()

        if self.rows > 1 and self._last_ts > self._first_ts:
            rate = (self.rows - 1) / (self._last_ts - self._first_ts)
            logging.info(f"Записано {self.rows} рядків. Середня частота: {rate:.1f} Гц.")
        logging.info("Файл телеметрії збережено.")

# ============================================================================
# ГОЛОВНИЙ КЛАС (APP)
# ============================================================================
class MotorApp:
    def __init__(self, port=None, scenario_file=None):
        self.serial_mgr = SerialManager(Config.BAUD_RATE, port)
        self.logger = DataLogger("telemetry", Config.LOG_WITH_TIMESTAMP)
        self.stop_event = threading.Event()
        self.plot_queue = queue.Queue()
        self.scenario_file = scenario_file

        self.t_data = deque(maxlen=Config.MAX_POINTS)
        self.w_data = deque(maxlen=Config.MAX_POINTS)
        self.tgt_data = deque(maxlen=Config.MAX_POINTS)
        self.pwm_data = deque(maxlen=Config.MAX_POINTS)
        self.t0 = None
        self.reader = None
        self.ani = None

        # Прапорець для безпечного cleanup() з будь-якого потоку
        self._cleaned = False
        self._cleanup_lock = threading.Lock()

        self._setup_command_logger()

    def _setup_command_logger(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self.cmd_logger = logging.getLogger("commands")
        self.cmd_logger.setLevel(logging.INFO)
        self.cmd_logger.propagate = False
        handler = logging.handlers.RotatingFileHandler(
            os.path.join(Config.LOG_DIR, "commands.log"),
            maxBytes=1_048_576, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        self.cmd_logger.addHandler(handler)

    def _on_sample(self, ts, data):
        if self.t0 is None: self.t0 = ts
        self.logger.log(data, ts)
        if Config.ENABLE_PLOT:
            # Перші 4 поля — для графіка
            self.plot_queue.put((ts - self.t0, data[:4]))

    def cli_thread(self):
        logging.info("Введіть команду (наприклад, 'w 15', 'f 100', 's', 'status', 'diag'):")
        while not self.stop_event.is_set():
            try:
                cmd = input()
                if cmd.strip():
                    self.serial_mgr.send(cmd.strip())
                    self.cmd_logger.info(cmd.strip())
            except (KeyboardInterrupt, EOFError):
                self.cleanup()
                break

    def scenario_thread(self, filename):
        if not os.path.exists(filename):
            logging.error(f"Файл сценарію '{filename}' не знайдено.")
            self.cleanup()
            return

        logging.info(f"Запуск сценарію з файлу: {filename}")

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.lower().startswith("wait "):
                    try:
                        delay = float(line.split(" ")[1])
                        logging.info(f"[СЦЕНАРІЙ] Пауза {delay} сек...")
                        if self.stop_event.wait(delay):
                            break
                    except (ValueError, IndexError):
                        logging.error(f"Помилка парсингу паузи: {line}")
                    continue

                if line.lower() == "stop_all":
                    logging.info("[СЦЕНАРІЙ] Завершення сценарію.")
                    break

                self.serial_mgr.send(line)
                self.cmd_logger.info(line)
                time.sleep(0.05)

        logging.info("Сценарій завершено.")
        self.cleanup()

    def run(self):
        if not self.serial_mgr.connect():
            self.logger.close()
            return

        self.reader = threading.Thread(
            target=self.serial_mgr.read_loop, args=(self.stop_event, self._on_sample), daemon=True)
        self.reader.start()

        if self.scenario_file:
            threading.Thread(target=self.scenario_thread, args=(self.scenario_file,), daemon=True).start()
        else:
            threading.Thread(target=self.cli_thread, daemon=True).start()

        if Config.ENABLE_PLOT:
            self._setup_plot()
        else:
            try:
                while not self.stop_event.is_set():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                self.cleanup()

    def _setup_plot(self):
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        self.fig.canvas.manager.set_window_title('PBC Motor Telemetry (Auto Scenario)')
        self.line_w, = self.ax1.plot([], [], 'c-', linewidth=1.5, label='Actual ω (rad/s)')
        self.line_tgt, = self.ax1.plot([], [], 'r--', linewidth=1.5, label='Target ω* (rad/s)')
        self.ax1.set_ylabel('Speed (rad/s)')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True, linestyle=':', alpha=0.4)

        self.line_pwm, = self.ax2.plot([], [], 'y-', linewidth=1.5, label='PWM (0-255)')
        self.ax2.set_ylabel('PWM Duty Cycle')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylim(-5, 260)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, linestyle=':', alpha=0.4)

        plt.tight_layout()
        try:
            self.ani = animation.FuncAnimation(
                self.fig, self._update, interval=50, blit=False, cache_frame_data=False)
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def _update(self, frame):
        while True:
            try:
                t, (current, omega, target, pwm) = self.plot_queue.get_nowait()
            except queue.Empty:
                break
            self.t_data.append(t)
            self.w_data.append(omega)
            self.tgt_data.append(target)
            self.pwm_data.append(pwm)

        if not self.t_data:
            return self.line_w, self.line_tgt, self.line_pwm

        self.line_w.set_data(self.t_data, self.w_data)
        self.line_tgt.set_data(self.t_data, self.tgt_data)
        self.line_pwm.set_data(self.t_data, self.pwm_data)

        self.ax1.set_xlim(max(0, self.t_data[0]), max(5, self.t_data[-1]))
        lo = min(min(self.w_data), min(self.tgt_data), 0.0)
        hi = max(max(self.w_data), max(self.tgt_data), 0.0)
        margin = 0.1 * max(hi - lo, 5.0)
        self.ax1.set_ylim(lo - margin, hi + margin)

        return self.line_w, self.line_tgt, self.line_pwm

    def cleanup(self):
        # Захист від повторного виклику з кількох потоків
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True

        logging.info("Завершення роботи...")
        try:
            self.serial_mgr.send("s")
        except Exception as e:
            logging.warning(f"Не вдалося надіслати 's': {e}")
        time.sleep(0.1)
        self.stop_event.set()

        if self.reader:
            self.reader.join(timeout=1.0)
        self.serial_mgr.close()
        self.logger.close()
        logging.info("Усі ресурси успішно звільнено.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PBC Motor Control & Telemetry")
    parser.add_argument("port", nargs="?", default=None, help="COM-порт (наприклад, COM3)")
    parser.add_argument("--scenario", "-s", type=str, help="Шлях до файлу зі сценарієм команд")
    args = parser.parse_args()

    app = MotorApp(port=args.port, scenario_file=args.scenario)
    app.run()