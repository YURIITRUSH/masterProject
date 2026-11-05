import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from scipy.optimize import differential_evolution
from core.digital_twin import DigitalTwin

def main():
    targets = [5.0, 10.0, 15.0] # ШІ знайде ідеальні Kp, Ki, Kd для кожної
    optimal_table = {}
    
    twin = DigitalTwin()
    bounds = [(0.1, 1.5), (0.01, 1.0), (0.0001, 0.02)]

    for target_speed in targets:
        print(f"🤖 ШІ шукає для {target_speed} рад/с...")
        
        def objective(gains):
            iae, max_overshoot = twin.simulate_closed_loop_with_overshoot(target_speed, *gains)
            penalty = (max_overshoot - 15.0) * 500.0 if max_overshoot > 15.0 else 0.0
            return iae + penalty

        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=30, popsize=10)
        kp, ki, kd = result.x
        print(f"  -> Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.5f}")
        
        optimal_table[str(target_speed)] = {"Kp": kp, "Ki": ki, "Kd": kd}
        
    # Зберігаємо повну таблицю
    with open('ai/optimal_gains_table.json', 'w') as f:
        json.dump(optimal_table, f, indent=4)
    print("💾 Збережено таблицю у ai/optimal_gains_table.json")
    
    best = optimal_table["15.0"]
    with open('ai/optimal_gains.json', 'w') as f:
        json.dump({"Kp": best["Kp"], "Ki": best["Ki"], "Kd": best["Kd"], "target": 15.0}, f)
    print("💾 Збережено таблицю у ai/optimal_gains.json")

if __name__ == "__main__":
    main()