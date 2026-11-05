import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    t = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds().values
    return t, df

def compare(file_old, file_new):
    t_old, df_old = load_data(file_old)
    t_new, df_new = load_data(file_new)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Порівняння: Ручний тюнінг vs ШІ-тюнінг', fontsize=16, fontweight='bold')

    # 1. Швидкість
    ax1.plot(t_old, df_old['Omega'], 'b-', alpha=0.7, label='Ручний підбір (Old)')
    ax1.plot(t_new, df_new['Omega'], 'r-', linewidth=2, label='Оптимізовано ШІ (New)')
    ax1.plot(t_old, df_old['Target'], 'k--', alpha=0.5, label='Ціль')
    ax1.set_ylabel('Швидкість (рад/с)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. ШІМ
    ax2.plot(t_old, df_old['PWM'], 'b-', alpha=0.7, label='Ручний підбір (Old)')
    ax2.plot(t_new, df_new['PWM'], 'g-', linewidth=2, label='Оптимізовано ШІ (New)')
    ax2.set_ylabel('PWM (0-255)')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Струм
    ax3.plot(t_old, df_old['Current_A'], 'b-', alpha=0.7, label='Ручний підбір (Old)')
    ax3.plot(t_new, df_new['Current_A'], 'm-', linewidth=2, label='Оптимізовано ШІ (New)')
    ax3.set_ylabel('Струм (А)')
    ax3.set_xlabel('Час (с)')
    ax3.legend()
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    
    # Розрахунок метрик (простий IAE для наочності)
    # Знаходимо момент досягнення 15 рад/с
    idx_old = np.where(df_old['Omega'] >= 14.5)[0]
    idx_new = np.where(df_new['Omega'] >= 14.5)[0]
    
    if len(idx_old) > 0:
        print(f"⏱️ Час виходу на ціль (Old): {t_old[idx_old[0]]:.2f} с")
    if len(idx_new) > 0:
        print(f"⏱️ Час виходу на ціль (New): {t_new[idx_new[0]]:.2f} с")

    plt.show()

if __name__ == "__main__":
    # Вкажіть назви ваших файлів
    compare('logs/test_old_gains.csv', 'logs/test_ai_gains.csv')