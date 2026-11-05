import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import json
from scipy.optimize import differential_evolution

def find_latest_csv():
    files = [os.path.join('logs', f) for f in os.listdir('logs') if f.endswith('.csv')]
    return max(files, key=os.path.getmtime) if files else None

def simulate_euler(params, t_span, u_history):
    R, L, Ke, Km, J, b, Ts = params
    dt = 0.005 # Фіксований крок 5 мс (200 Гц)
    
    w_sim = np.zeros(len(t_span))
    i = 0.0
    w = 0.0
    
    for k in range(len(t_span) - 1):
        u = u_history[k]
        
        # Електрика
        di_dt = (u - R * i - Ke * w) / L
        i_next = i + di_dt * dt
        
        # Механіка
        torque = Km * i
        if abs(w) < 0.1:
            if abs(torque) < Ts:
                dw_dt = 0.0
            else:
                dw_dt = (torque - Ts * np.sign(torque)) / J
        else:
            dw_dt = (torque - Ts * np.sign(w) - b * w) / J
        
        w = w + dw_dt * dt
        i = i_next
        
        # ЗАХИСТ ВІД ВИБУХУ (якщо струм або швидкість стають нереальними)
        if abs(i) > 20 or abs(w) > 100:
            return np.full(len(t_span), np.nan)
            
        w_sim[k+1] = w
        
    return w_sim

def main():
    csv_file = find_latest_csv()
    if not csv_file:
        print("Помилка: CSV файл не знайдено.")
        return

    print(f"📊 Калібрування за файлом: {csv_file}")
    df = pd.read_csv(csv_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    t_real = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds().values
    
    # Беремо тільки перші 4 секунди (де відбувається розгін)
    mask = t_real <= 4.0
    t_real = t_real[mask]
    V_supply = 11.2
    u_input = (df['PWM'].values[mask] / 255.0) * V_supply
    w_real = df['Omega'].values[mask]

    if len(t_real) < 10:
        print("Помилка: лог занадто короткий.")
        return

    # Робимо час РІВНОМІРНИМ (крок 5 мс) для стабільності Ейлера
    t_uniform = np.arange(t_real[0], t_real[-1], 0.005)
    u_uniform = np.interp(t_uniform, t_real, u_input)
    w_real_uniform = np.interp(t_uniform, t_real, w_real)

    # Адекватні фізичні межі для мотора N20 (12V)
    bounds = [
        (2.0, 10.0),    # R (Опір, Ом)
        (0.0005, 0.01), # L (Індуктивність, Гн)
        (0.1, 1.2),     # Ke (Проти-ЕРС, В/(рад/с))
        (0.1, 1.2),     # Km (Момент, Н*м/А)
        (1e-6, 0.001),  # J (Інерція, кг*м^2)
        (1e-6, 0.005),  # b (В'язке тертя)
        (0.0, 0.15)     # Ts (Сухе тертя)
    ]

    iteration_count = [0]

    def objective(params):
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            print(f"  ... перевірено {iteration_count[0]} комбінацій ...")
            
        w_sim = simulate_euler(params, t_uniform, u_uniform)
        
        # Якщо модель вибухнула — штраф
        if np.isnan(w_sim).any() or np.isinf(w_sim).any():
            return 1e6
            
        return np.sqrt(np.mean((w_real_uniform - w_sim) ** 2))

    print("🤖 ШІ шукає фізичні параметри (це займе ~20 секунд)...")
    
    result = differential_evolution(
        objective, bounds, strategy='best1bin', maxiter=20, popsize=15, tol=0.01
    )
    
    if np.isnan(result.fun) or result.fun >= 1e6:
        print("❌ Не вдалося знайти стабільні параметри. Спробуйте інший лог.")
        return

    best_params = result.x
    
    print(f"\n✅ Калібрування завершено! Похибка (RMSE): {result.fun:.3f} рад/с")
    print(f"  R  = {best_params[0]:.3f} Ом")
    print(f"  L  = {best_params[1]:.5f} Гн")
    print(f"  Ke = {best_params[2]:.3f} В/(рад/с)")
    print(f"  Km = {best_params[3]:.3f} Н*м/А")
    print(f"  J  = {best_params[4]:.7f} кг*м^2")
    print(f"  b  = {best_params[5]:.7f}")
    print(f"  Ts = {best_params[6]:.4f}")

    # Оновлюємо config.json
    config_path = os.path.join(os.path.dirname(__file__), '..', 'ai', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    config['motor_parameters'] = {
        "R": float(best_params[0]), "L": float(best_params[1]), "Ke": float(best_params[2]),
        "Km": float(best_params[3]), "J": float(best_params[4]), "b": float(best_params[5]),
        "Ts": float(best_params[6])
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print("\n💾 Оновлено ai/config.json з новими параметрами!")

if __name__ == "__main__":
    main()