import serial
import serial.tools.list_ports
import time
import numpy as np
import tensorflow as tf
import pickle
import os
import csv
from datetime import datetime

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in p.description.lower() for k in ["arduino", "ch340", "cp210", "usb"]):
            return p.device
    return ports[0].device if ports else None

def main():
    # 1. Завантажуємо модель
    model = tf.keras.models.load_model('ai/tf_model.keras')
    with open('ai/tf_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    port = find_arduino_port()
    if not port:
        print("Помилка: Arduino не знайдено.")
        return
        
    # 2. Створюємо файл для логування
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(f'logs/tf_telemetry_{timestamp}.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["Timestamp", "Current_A", "Omega", "Target", "PWM", "Kp", "Ki", "Kd"])
    
    ser = serial.Serial(port, 115200, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()
    
    target_speed = 8.0
    print(f"🚀 Запуск з нейромережею на цільовій швидкості {target_speed} рад/с...")
    ser.write(f"cfreq 100\n".encode())
    ser.write(f"w {target_speed}\n".encode())
    
    last_tune_time = time.time()
    current_kp, current_ki, current_kd = 0.4, 0.1, 0.002 # Початкові
    
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 4:
                    current = float(parts[0])
                    omega = float(parts[1])
                    target = float(parts[2])
                    pwm = float(parts[3])
                    
                    # Раз на 100 мс оновлюємо коефіцієнти
                    if time.time() - last_tune_time > 0.1 and target != 0:
                        error = target - omega
                        
                        X = np.array([[target, error]])
                        X_norm = (X - scaler['X_mean']) / scaler['X_std']
                        Y_norm = model.predict(X_norm, verbose=0)
                        Y = Y_norm * scaler['Y_std'] + scaler['Y_mean']
                        
                        current_kp, current_ki, current_kd = np.clip(Y[0], [0.1, 0.01, 0.0001], [1.5, 1.0, 0.02])
                        
                        ser.write(f"kp {current_kp:.3f}\n".encode())
                        ser.write(f"ki {current_ki:.3f}\n".encode())
                        ser.write(f"kd {current_kd:.5f}\n".encode())
                        
                        print(f"Target: {target:5.1f} | Error: {error:6.2f} | Predicted: Kp={current_kp:.3f}, Ki={current_ki:.3f}, Kd={current_kd:.5f}")
                        last_tune_time = time.time()
                    
                    # Логуємо кожен семпл телеметрії
                    writer.writerow([datetime.now().isoformat(), current, omega, target, pwm, f"{current_kp:.4f}", f"{current_ki:.4f}", f"{current_kd:.5f}"])
                    log_file.flush()
                        
    except KeyboardInterrupt:
        print("\n🛑 Зупинка...")
    finally:
        ser.write(b"s\n")
        ser.close()
        log_file.close()
        print(f"💾 Лог збережено у logs/tf_telemetry_{timestamp}.csv")

if __name__ == "__main__":
    main()