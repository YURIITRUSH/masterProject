import serial
import serial.tools.list_ports
import json
import time
import os

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(k in p.description.lower() for k in ["arduino", "ch340", "cp210", "usb"]):
            return p.device
    return ports[0].device if ports else None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gains_file = os.path.join(script_dir, 'optimal_gains.json')

    if not os.path.exists(gains_file):
        print(f"Помилка: не знайдено {gains_file}. Спочатку запустіть ai_tuning.py")
        return

    with open(gains_file, 'r') as f:
        data = json.load(f)
        
    port = find_arduino_port()
    if not port:
        print("Помилка: Arduino не знайдено.")
        return
        
    print(f"🔌 Підключення до {port}...")
    ser = None
    try:
        ser = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(2) 
        
        # Відправляємо коефіцієнти
        ser.write(f"kp {data['Kp']}\n".encode()); time.sleep(0.1)
        ser.write(f"ki {data['Ki']}\n".encode()); time.sleep(0.1)
        ser.write(f"kd {data['Kd']}\n".encode()); time.sleep(0.1)
        
        print(f"Запуск мотора на {data['target']} рад/с...")
        ser.write(f"w {data['target']}\n".encode())
        ser.reset_input_buffer()
        
        print("Збір телеметрії (4 сек)...")
        start_time = time.time()
        omegas = []
        targets = []
        
        while time.time() - start_time < 4.0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line and ',' in line:
                try:
                    parts = line.split(',')
                    omegas.append(float(parts[1]))
                    targets.append(float(parts[2]))
                except:
                    pass
        
        ser.write(b"s\n")
        
        if omegas:
            final_omega = sum(omegas[-10:]) / 10.0
            target = targets[-1]
            error = abs(target - final_omega)
            print(f"\n📊 РЕЗУЛЬТАТ:")
            print(f"   Ціль: {target:.2f} рад/с")
            print(f"   Фактична швидкість: {final_omega:.2f} рад/с")
            print(f"   Похибка: {error:.2f} рад/с")
            if error < 0.5:
                print("ШІ впорався відмінно!")
            else:
                print("Є статична похибка.")

    except Exception as e:
        print(f"Помилка: {e}")
    finally:
        if ser and ser.is_open:
            try: ser.write(b"s\n")
            except: pass
            ser.close()
            print("Мотор зупинено, порт закрито.")

if __name__ == "__main__":
    main()