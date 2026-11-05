import subprocess
import os

def run_script(path):
    print(f"\n{'='*50}\n🚀 ЗАПУСК: {path}\n{'='*50}")
    # Запускаємо з кореневої папки
    result = subprocess.run(["python", path])
    if result.returncode != 0:
        print(f"❌ Помилка у {path}. Зупинка.")
        exit(1)

def main():
    print("ПОЧАТОК ПОВНОГО ЦИКЛУ HIL-ТЕСТУВАННЯ")
    
    # 1. ШІ шукає оптимальні Kp, Ki, Kd
    run_script("ai/ai_tuning.py")
    
    # 2. Застосування на реальному моторі (зупиниться автоматично)
    run_script("ai/apply_tuning.py")
    
    # 3. Аналіз останнього логу
    run_script("analysis/analyze.py")
    
    print("\n✅ ЦИКЛ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()