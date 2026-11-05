import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from core.digital_twin import DigitalTwin

def find_latest_csv():
    files = [os.path.join('logs', f) for f in os.listdir('logs') if f.endswith('.csv')]
    return max(files, key=os.path.getmtime) if files else None

def main():
    csv_file = find_latest_csv()
    if not csv_file:
        print("Помилка: У папці logs немає CSV файлів.")
        return

    print(f"📊 Аналізуємо файл: {csv_file}")
    
    df = pd.read_csv(csv_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    t_real = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds().values
    
    twin = DigitalTwin()
    
    # 1. Переводимо PWM у Вольти
    u_raw = (df['PWM'].values / 255.0) * twin.V_supply
    
    # 2. ЗГЛАДЖУЄМО СИГНАЛ (імітуємо інерційність L298N)
    # Беремо ковзне середнє по 10 точках (при 100 Гц це 0.1 сек)
    u_smooth = uniform_filter1d(u_raw, size=10)
    
    # 3. Симуляція двійника
    i_sim, w_sim = twin.simulate_ode(t_real, u_smooth)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(t_real, df['Omega'], 'b-', label='Реальний мотор', alpha=0.8)
    ax1.plot(t_real, w_sim, 'r--', label='Digital Twin')
    ax1.set_ylabel('Швидкість (рад/с)'); ax1.legend(); ax1.grid(True)
    
    ax2.plot(t_real, df['Current_A'], 'b-', label='Реальний мотор', alpha=0.8)
    ax2.plot(t_real, i_sim, 'r--', label='Digital Twin')
    ax2.set_ylabel('Струм (А)'); ax2.set_xlabel('Час (с)'); ax2.legend(); ax2.grid(True)
    
    # Рахуємо похибку
    rmse = np.sqrt(np.mean((df['Omega'] - w_sim)**2))
    print(f"✅ Похибка моделі (RMSE): {rmse:.3f} рад/с")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()