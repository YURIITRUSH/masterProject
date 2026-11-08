import serial
import time

PORT = "COM5" # ЗАМІНІТЬ НА ВАШ ПОРТ, ЯКЩО ТРЕБА
BAUD = 115200

print(f"🔌 Спроба підключення до {PORT}...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(5) # Чекаємо, поки Arduino прокинешся
    ser.reset_input_buffer()
    print("✅ Порт відкрито!")
    
    # 1. Перевірка зв'язку
    print("\n1. Відправляю 'status' (перевірка зв'язку)...")
    ser.write(b'status\n')
    time.sleep(1)
    response = ser.read_all().decode('utf-8', errors='ignore')
    if "STATUS" in response:
        print("✅ Arduino відповів! Зв'язок Є.")
        print(response[:200] + "...") # Показуємо початок відповіді
    else:
        print("❌ Arduino НЕ відповів на status. Ось що прийшло:")
        print(f"'{response}'")

    # 2. Спроба запустити мотор
    print("\n2. Відправляю 'f 100' (мотор має крутитися 3 секунди)...")
    ser.write(b'f 100\n')
    time.sleep(3) # Чекаємо 3 секунди
    
    # 3. Зупинка
    print("3. Відправляю 's' (стоп)...")
    ser.write(b's\n')
    time.sleep(0.5)
    
    # Читаємо логи, які Arduino встиг відправити
    final_logs = ser.read_all().decode('utf-8', errors='ignore')
    print("\n--- ЛОГИ ВІД ARDUINO ---")
    if final_logs:
        print(final_logs)
    else:
        print("(Порожньо. Arduino нічого не надіслав у відповідь).")
    
    ser.close()
    print("\n🔒 Порт закрито.")

except Exception as e:
    print(f"\n❌ Помилка: {e}")