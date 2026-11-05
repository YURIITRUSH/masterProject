import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.optimize import differential_evolution
from core.digital_twin import DigitalTwin
import pickle

def generate_dataset():
    print("Генерація даних для навчання нейромережі")
    twin = DigitalTwin()
    bounds = [(0.1, 1.5), (0.01, 1.0), (0.0001, 0.02)]
    
    X_data = []
    Y_data = []
    
    # Генеруємо дані для швидкостей від 3 до 18 рад/с
    targets = np.linspace(3.0, 18.0, 16)
    
    for i, target in enumerate(targets):
        print(f"Оптимізація для {target:.1f} рад/с ({i+1}/{len(targets)})")
        
        def objective(gains):
            iae, max_over = twin.simulate_closed_loop_with_overshoot(target, *gains)
            penalty = (max_over - 15.0) * 500.0 if max_over > 15.0 else 0.0
            return iae + penalty

        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=10, popsize=8)
        
        # Додаємо трохи "шуму" навколо цілі
        for _ in range(50):
            err = np.random.uniform(-2.0, 2.0)
            X_data.append([target, err])
            Y_data.append(result.x)
            
    return np.array(X_data), np.array(Y_data)

def train_model():
    X, Y = generate_dataset()
    print(f"Згенеровано {len(X)} прикладів.")
    
    # Нормалізація даних
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std
    
    # Створення моделі (2 входи -> 32 -> 16 -> 3 виходи)
    model = models.Sequential([
        layers.Input(shape=(2,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    print("Навчання нейромережі")
    model.fit(X_norm, Y_norm, epochs=50, batch_size=16, verbose=1)
    
    # Зберігаємо модель та параметри нормалізації
    model.save('ai/tf_model.keras')
    with open('ai/tf_scaler.pkl', 'wb') as f:
        pickle.dump({'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std}, f)
        
    print("Модель збережена у ai/tf_model.keras")

if __name__ == "__main__":
    train_model()