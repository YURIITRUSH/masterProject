import json
import os
import numpy as np
from scipy.integrate import solve_ivp

class DigitalTwin:
    def __init__(self, config_path=None):
        # Якщо шлях не вказано, шукаємо config.json у папці ai
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, '..', 'ai', 'config.json')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        m = self.config['motor_parameters']
        self.R, self.L, self.Ke = m['R'], m['L'], m['Ke']
        self.Km, self.J, self.b, self.Ts = m['Km'], m['J'], m['b'], m['Ts']
        self.V_supply = self.config['hardware']['supply_voltage']

    def _physics_step(self, u, i, omega, dt):
        di_dt = (u - self.R * i - self.Ke * omega) / self.L
        i_next = i + di_dt * dt

        torque = self.Km * i
        if abs(omega) < 0.5:
            if abs(torque) < self.Ts:
                dw_dt = 0.0
            else:
                dw_dt = (torque - self.Ts * np.sign(torque)) / self.J
        else:
            dw_dt = (torque - self.Ts * np.sign(omega) - self.b * omega) / self.J
        
        omega_next = omega + dw_dt * dt

        # ЗАХИСТ ВІД ВИБУХУ: Обмежуємо фізичні величини
        if np.isnan(omega_next) or np.isinf(omega_next) or abs(omega_next) > 200:
            omega_next = 200.0 * np.sign(omega_next) if not np.isnan(omega_next) else 0.0
        
        if np.isnan(i_next) or np.isinf(i_next) or abs(i_next) > 10:
            i_next = 10.0 * np.sign(i_next) if not np.isnan(i_next) else 0.0

        return i_next, omega_next

    def simulate_ode(self, t_span, u_history):
        import numpy as np

        R  = float(self.R)
        L  = float(self.L)
        Ke = float(self.Ke)
        Km = float(self.Km)
        J  = float(self.J)
        b  = float(self.b)
        Ts = float(getattr(self, 'Ts', 0.0))

        t_span    = np.asarray(t_span,    dtype=float)
        u_history = np.asarray(u_history, dtype=float)
        n = len(t_span)

        i_out = np.zeros(n)
        w_out = np.zeros(n)

        i = 0.0
        w = 0.0
        tau_e = L / max(R, 1e-9)
        max_internal_dt = 0.0005  # 0.5 мс

        for k in range(1, n):
            dt_total = t_span[k] - t_span[k-1]
            if dt_total <= 0:
                i_out[k] = i
                w_out[k] = w
                continue

            u = u_history[k-1]
            n_sub = max(1, int(np.ceil(dt_total / max_internal_dt)))
            dt = dt_total / n_sub

            for _ in range(n_sub):
                # ── 1. Струм (аналітично, точно для постійних u,w на кроці) ──
                # L di/dt = u - R*i - Ke*w
                u_eq  = u - Ke * w
                i_inf = u_eq / max(R, 1e-9)
                i = i_inf + (i - i_inf) * np.exp(-dt / max(tau_e, 1e-9))

                # ── 2. Механіка (Ейлер) ──
                # J dw/dt = Km*i - b*w - Ts*sign(w)
                T_motor = Km * i
                T_visc  = b * w
                T_net   = T_motor - T_visc

                if abs(w) < 1e-4:
                    if abs(T_net) < Ts:
                        w = 0.0
                        continue
                    T_coulomb = Ts * np.sign(T_net)
                else:
                    T_coulomb = Ts * np.sign(w)

                dw = (T_net - T_coulomb) / max(J, 1e-12)
                w += dw * dt

            i_out[k] = i
            w_out[k] = w

        return i_out, w_out

    def simulate_closed_loop(self, target, kp, ki, kd, dt=0.01, T=4.0):
        steps = int(T / dt)
        i, w = 0.0, 0.0
        e_sum, prev_e = 0.0, target - w
        iae = 0.0 

        for _ in range(steps):
            e = target - w
            e_sum = np.clip(e_sum + ki * e * dt, -2.0, 2.0)
            D = kd * (e - prev_e) / dt
            prev_e = e
            i_ref = e_sum + kp * e + D
            
            u = self.Ke * target + self.R * i_ref
            u = np.clip(u, -self.V_supply, self.V_supply)
            
            i, w = self._physics_step(u, i, w, dt)
            
            # ДОДАЙТЕ ЦЕ: Якщо модель "вибухнула", вертаємо великий штраф
            if np.isnan(w) or np.isinf(w) or np.isnan(i) or np.isinf(i):
                return 1e6
                
            iae += abs(e) * dt
        return iae
    
    def simulate_closed_loop_with_overshoot(self, target, kp, ki, kd, dt=0.01, T=4.0):
        steps = int(T / dt)
        i, w = 0.0, 0.0
        e_sum, prev_e = 0.0, target - w
        iae = 0.0 
        max_w = 0.0

        for _ in range(steps):
            e = target - w
            e_sum = np.clip(e_sum + ki * e * dt, -2.0, 2.0)
            D = kd * (e - prev_e) / dt
            prev_e = e
            i_ref = e_sum + kp * e + D
            
            u = self.Ke * target + self.R * i_ref
            u = np.clip(u, -self.V_supply, self.V_supply)
            
            i, w = self._physics_step(u, i, w, dt)
            
            if w > max_w:
                max_w = w
                
            iae += abs(e) * dt
            
        overshoot = ((max_w - target) / target) * 100.0 if target > 0 else 0.0
        return iae, max(overshoot, 0.0)