import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os
from scipy.signal import find_peaks

class QuadrotorPIDOptimizer:
    def __init__(self):
        self.m = 1.0
        self.g = 9.81
        self.Ix = 0.1
        self.Iy = 0.1
        self.Iz = 0.2
        self.t_span = (0, 20)
        
    def ziegler_nichols_tuning(self, z_des, phi_des, theta_des, psi_des):
        """
        Método Ziegler-Nichols mejorado para sintonización PID
        """
        print("=== SINTONIZACIÓN ZIEGLER-NICHOLS ===")
        
        # Búsqueda de Ku (ganancia última) y Pu (período último)
        Ku_z, Pu_z = self.find_ultimate_gain(z_des, phi_des, theta_des, psi_des, 'z')
        Ku_phi, Pu_phi = self.find_ultimate_gain(z_des, phi_des, theta_des, psi_des, 'phi')
        Ku_theta, Pu_theta = self.find_ultimate_gain(z_des, phi_des, theta_des, psi_des, 'theta')
        Ku_psi, Pu_psi = self.find_ultimate_gain(z_des, phi_des, theta_des, psi_des, 'psi')
        
        # Cálculo de parámetros PID según Ziegler-Nichols
        if Ku_z and Pu_z:
            Kp_z = 0.6 * Ku_z
            Ki_z = 1.2 * Ku_z / Pu_z
            Kd_z = 0.075 * Ku_z * Pu_z
            print(f"Altitud (z): Kp={Kp_z:.3f}, Ki={Ki_z:.3f}, Kd={Kd_z:.3f}")
        else:
            # Valores por defecto si no se encuentra Ku
            Kp_z, Ki_z, Kd_z = 5.0, 0.5, 1.0
            
        if Ku_phi and Pu_phi:
            Kp_phi = 0.6 * Ku_phi
            Ki_phi = 1.2 * Ku_phi / Pu_phi
            Kd_phi = 0.075 * Ku_phi * Pu_phi
            print(f"Roll (ϕ): Kp={Kp_phi:.3f}, Ki={Ki_phi:.3f}, Kd={Kd_phi:.3f}")
        else:
            Kp_phi, Ki_phi, Kd_phi = 2.0, 0.1, 0.5
            
        if Ku_theta and Pu_theta:
            Kp_theta = 0.6 * Ku_theta
            Ki_theta = 1.2 * Ku_theta / Pu_theta
            Kd_theta = 0.075 * Ku_theta * Pu_theta
            print(f"Pitch (θ): Kp={Kp_theta:.3f}, Ki={Ki_theta:.3f}, Kd={Kd_theta:.3f}")
        else:
            Kp_theta, Ki_theta, Kd_theta = 2.0, 0.1, 0.5
            
        if Ku_psi and Pu_psi:
            Kp_psi = 0.6 * Ku_psi
            Ki_psi = 1.2 * Ku_psi / Pu_psi
            Kd_psi = 0.075 * Ku_psi * Pu_psi
            print(f"Yaw (ψ): Kp={Kp_psi:.3f}, Ki={Ki_psi:.3f}, Kd={Kd_psi:.3f}")
        else:
            Kp_psi, Ki_psi, Kd_psi = 1.0, 0.05, 0.2
        
        zn_gains = np.array([Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi, 
                           Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi])
        
        return zn_gains
    
    def find_ultimate_gain(self, z_des, phi_des, theta_des, psi_des, control_type):
        """
        Encuentra la ganancia última Ku y período último Pu para un tipo de control específico
        """
        Kp_values = np.linspace(0.1, 20, 100)
        Ku = None
        Pu = None
        
        for Kp_test in Kp_values:
            # Configurar ganancias según el tipo de control
            if control_type == 'z':
                gains = [Kp_test, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif control_type == 'phi':
                gains = [5.0, 0.5, 1.0, Kp_test, 0, 0, 0, 0, 0, 0, 0, 0]
            elif control_type == 'theta':
                gains = [5.0, 0.5, 1.0, 2.0, 0.1, 0.5, Kp_test, 0, 0, 0, 0, 0]
            else:  # psi
                gains = [5.0, 0.5, 1.0, 2.0, 0.1, 0.5, 2.0, 0.1, 0.5, Kp_test, 0, 0]
            
            try:
                # Simular sistema
                fitness, metrics, t, response = self.evaluate_pid(
                    gains, z_des, phi_des, theta_des, psi_des, disturbance=False)
                
                # Detectar oscilaciones sostenidas
                if len(t) > 100:
                    # Usar la respuesta correspondiente al tipo de control
                    if control_type == 'z':
                        signal = response
                    elif control_type == 'phi':
                        signal = self.get_phi_from_simulation(gains, z_des, phi_des, theta_des, psi_des)
                    elif control_type == 'theta':
                        signal = self.get_theta_from_simulation(gains, z_des, phi_des, theta_des, psi_des)
                    else:
                        signal = self.get_psi_from_simulation(gains, z_des, phi_des, theta_des, psi_des)
                    
                    # Encontrar picos para detectar oscilaciones
                    peaks, _ = find_peaks(np.abs(signal - signal[0]), height=0.01)
                    
                    if len(peaks) >= 3:  # Mínimo 3 picos para considerar oscilación sostenida
                        period = np.mean(np.diff(t[peaks]))
                        Ku = Kp_test
                        Pu = period
                        break
                        
            except:
                continue
                
        return Ku, Pu
    
    def get_phi_from_simulation(self, gains, z_des, phi_des, theta_des, psi_des):
        """Obtiene la respuesta en roll de una simulación"""
        x0 = np.zeros(6)
        xdot0 = np.zeros(6)
        X0 = np.concatenate((x0, xdot0))
        
        sol = solve_ivp(
            lambda t, X: self.quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, False),
            self.t_span, X0, t_eval=np.linspace(0, 10, 1000)
        )
        
        return sol.y[3, :]  # Roll
    
    def get_theta_from_simulation(self, gains, z_des, phi_des, theta_des, psi_des):
        """Obtiene la respuesta en pitch de una simulación"""
        x0 = np.zeros(6)
        xdot0 = np.zeros(6)
        X0 = np.concatenate((x0, xdot0))
        
        sol = solve_ivp(
            lambda t, X: self.quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, False),
            self.t_span, X0, t_eval=np.linspace(0, 10, 1000)
        )
        
        return sol.y[4, :]  # Pitch
    
    def get_psi_from_simulation(self, gains, z_des, phi_des, theta_des, psi_des):
        """Obtiene la respuesta en yaw de una simulación"""
        x0 = np.zeros(6)
        xdot0 = np.zeros(6)
        X0 = np.concatenate((x0, xdot0))
        
        sol = solve_ivp(
            lambda t, X: self.quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, False),
            self.t_span, X0, t_eval=np.linspace(0, 10, 1000)
        )
        
        return sol.y[5, :]  # Yaw
    
    def pso_optimization(self, z_des, phi_des, theta_des, psi_des, initial_gains=None, disturbance=False):
        """
        Optimización PSO para parámetros PID
        """
        print(f"\n=== OPTIMIZACIÓN PSO {'CON PERTURBACIÓN' if disturbance else 'SIN PERTURBACIÓN'} ===")
        
        nVar = 12
        if initial_gains is None:
            VarMin = np.array([2.0, 0.01, 0.1, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1, 0.1, 0.001, 0.1])
            VarMax = np.array([15, 2.0, 5.0, 10, 0.1, 2.0, 10, 0.1, 2.0, 10, 0.1, 2.0])
        else:
            # Usar initial_gains como centro, con ±50% de variación
            VarMin = initial_gains * 0.5
            VarMax = initial_gains * 1.5
            # Asegurar límites mínimos
            VarMin = np.maximum(VarMin, [0.1, 0.001, 0.01, 0.01, 0.0001, 0.01, 0.01, 0.0001, 0.01, 0.01, 0.0001, 0.01])
        
        MaxIter = 50
        nPop = 30
        w = 0.7
        d = 0.97
        c1 = 1.7
        c2 = 1.7
        
        # Inicializar partículas
        particles = []
        global_best = {'position': None, 'fitness': float('inf')}
        convergence = np.zeros(MaxIter)
        
        for i in range(nPop):
            if initial_gains is None:
                position = np.random.uniform(VarMin, VarMax)
            else:
                # Perturbación aleatoria alrededor de initial_gains
                position = initial_gains * np.random.uniform(0.8, 1.2, nVar)
                position = np.clip(position, VarMin, VarMax)
                
            particle = {
                'position': position,
                'velocity': np.zeros(nVar),
                'fitness': float('inf'),
                'best': {'position': position.copy(), 'fitness': float('inf')}
            }
            particle['fitness'], _, _, _ = self.evaluate_pid(
                particle['position'], z_des, phi_des, theta_des, psi_des, disturbance)
            particle['best']['fitness'] = particle['fitness']
            
            if particle['fitness'] < global_best['fitness']:
                global_best = {'position': particle['position'].copy(), 'fitness': particle['fitness']}
            
            particles.append(particle)
        
        # Bucle principal PSO
        for iteration in range(MaxIter):
            for i in range(nPop):
                r1 = np.random.rand(nVar)
                r2 = np.random.rand(nVar)
                
                # Actualizar velocidad
                particles[i]['velocity'] = (w * particles[i]['velocity'] + 
                    c1 * r1 * (particles[i]['best']['position'] - particles[i]['position']) + 
                    c2 * r2 * (global_best['position'] - particles[i]['position']))
                
                # Actualizar posición
                particles[i]['position'] = np.clip(
                    particles[i]['position'] + particles[i]['velocity'], VarMin, VarMax)
                
                # Evaluar nueva posición
                fitness, metrics, t, z = self.evaluate_pid(
                    particles[i]['position'], z_des, phi_des, theta_des, psi_des, disturbance)
                particles[i]['fitness'] = fitness
                
                # Actualizar mejor personal
                if fitness < particles[i]['best']['fitness']:
                    particles[i]['best']['position'] = particles[i]['position'].copy()
                    particles[i]['best']['fitness'] = fitness
                    
                    # Actualizar mejor global
                    if fitness < global_best['fitness']:
                        global_best = {'position': particles[i]['position'].copy(), 'fitness': fitness}
            
            # Actualizar peso de inercia
            w = max(w * d, 0.4)
            convergence[iteration] = global_best['fitness']
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1}: Mejor fitness = {global_best['fitness']:.6f}")
        
        # Evaluación final del mejor resultado
        final_fitness, final_metrics, t_best, z_best = self.evaluate_pid(
            global_best['position'], z_des, phi_des, theta_des, psi_des, disturbance)
        
        print(f"Optimización completada. Fitness final: {final_fitness:.6f}")
        
        return global_best, final_metrics, convergence, t_best, z_best
    
    def evaluate_pid(self, gains, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """
        Evalúa el desempeño del controlador PID con las ganancias dadas
        """
        x0 = np.zeros(6)
        xdot0 = np.zeros(6)
        X0 = np.concatenate((x0, xdot0))
        
        metrics = {
            't_settle': np.nan,
            'overshoot': np.nan,
            't_rise': np.nan,
            'steady_error': np.nan,
            'ITSE': np.nan,
            'IAE': np.nan,
            'RMSE': np.nan
        }
        
        try:
            # Resolver ODE
            sol = solve_ivp(
                lambda t, X: self.quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance),
                (0, 10), X0, t_eval=np.linspace(0, 10, 1000), method='RK45'
            )
            
            t = sol.t
            X = sol.y
            z = X[2, :]  # Altura
            
            error_z = z_des - z
            
            # Calcular métricas de desempeño
            tol = 0.02 * abs(z_des)
            idx_settle = np.where(np.abs(error_z) > tol)[0]
            metrics['t_settle'] = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
            
            metrics['overshoot'] = max(0, (np.max(z) - z_des) / abs(z_des) * 100) if z_des != 0 else 0
            
            # Tiempo de levantamiento
            try:
                rise_start = 0.1 * abs(z_des)
                rise_end = 0.9 * abs(z_des)
                t_rise_start = t[np.where(z >= (z_des - rise_start) if z_des >= 0 else z <= (z_des + rise_start))[0][0]]
                t_rise_end = t[np.where(z >= (z_des - rise_end) if z_des >= 0 else z <= (z_des + rise_end))[0][0]]
                metrics['t_rise'] = t_rise_end - t_rise_start
            except:
                metrics['t_rise'] = np.nan
            
            # Error en estado estacionario
            metrics['steady_error'] = np.mean(np.abs(error_z[-100:]))
            
            # Métricas integrales
            metrics['ITSE'] = np.trapz(t * error_z**2, t)
            metrics['IAE'] = np.trapz(np.abs(error_z), t)
            metrics['RMSE'] = np.sqrt(np.mean(error_z**2))
            
            # Función de fitness (ponderada)
            fitness = (0.25 * min(metrics['t_settle']/5, 1) + 
                      0.25 * min(metrics['overshoot']/50, 1) + 
                      0.25 * min(metrics['ITSE']/10, 1) + 
                      0.25 * min(metrics['IAE']/5, 1))
            
        except Exception as e:
            # Si falla la simulación, retornar valores altos
            fitness = 1000
            t = np.linspace(0, 10, 100)
            z = np.zeros_like(t)
            metrics = {k: 1000 for k in metrics}
        
        return fitness, metrics, t, z
    
    def quadrotor_dynamics(self, t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """
        Dinámica del cuadróptero con control PID
        """
        # Inicializar variables persistentes para términos integrales
        if not hasattr(self, 'integral_vars'):
            self.integral_vars = {
                'iz': 0, 'ip': 0, 'it': 0, 'ipsi': 0,
                'prev_t': 0
            }
        
        # Extraer ganancias
        Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi, \
        Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi = gains
        
        pos = X[:6]   # [x, y, z, ϕ, θ, ψ]
        vel = X[6:]   # [vx, vy, vz, ωϕ, ωθ, ωψ]
        
        # Calcular errores
        errors = np.array([
            z_des - pos[2],    # Error en z
            phi_des - pos[3],  # Error en ϕ
            theta_des - pos[4], # Error en θ
            psi_des - pos[5]   # Error en ψ
        ])
        
        # Calcular dt
        dt = t - self.integral_vars['prev_t']
        self.integral_vars['prev_t'] = t
        
        # Actualizar términos integrales (con anti-windup)
        max_int = 10
        self.integral_vars['iz'] = np.clip(self.integral_vars['iz'] + errors[0] * dt, -max_int, max_int)
        self.integral_vars['ip'] = np.clip(self.integral_vars['ip'] + errors[1] * dt, -max_int, max_int)
        self.integral_vars['it'] = np.clip(self.integral_vars['it'] + errors[2] * dt, -max_int, max_int)
        self.integral_vars['ipsi'] = np.clip(self.integral_vars['ipsi'] + errors[3] * dt, -max_int, max_int)
        
        # Calcular señales de control
        U1 = Kp_z * errors[0] + Ki_z * self.integral_vars['iz'] + Kd_z * (-vel[2])
        U2 = Kp_phi * errors[1] + Ki_phi * self.integral_vars['ip'] + Kd_phi * (-vel[3])
        U3 = Kp_theta * errors[2] + Ki_theta * self.integral_vars['it'] + Kd_theta * (-vel[4])
        U4 = Kp_psi * errors[3] + Ki_psi * self.integral_vars['ipsi'] + Kd_psi * (-vel[5])
        
        # Perturbación de viento (si está activada)
        F_d = np.zeros(3)
        if disturbance:
            F_d = np.array([
                0.5 * np.sin(0.5 * t),    # Fx
                0.5 * np.cos(0.5 * t),    # Fy
                0.1 * np.sin(0.3 * t)     # Fz (pequeña perturbación vertical)
            ])
        
        # Aceleraciones lineales
        acc_lin = np.array([
            (np.cos(pos[3]) * np.sin(pos[4]) * np.cos(pos[5]) + np.sin(pos[3]) * np.sin(pos[5])) * U1 / self.m + F_d[0]/self.m,
            (np.cos(pos[3]) * np.sin(pos[4]) * np.sin(pos[5]) - np.sin(pos[3]) * np.cos(pos[5])) * U1 / self.m + F_d[1]/self.m,
            (np.cos(pos[3]) * np.cos(pos[4]) * U1 / self.m) - self.g + F_d[2]/self.m
        ])
        
        # Aceleraciones angulares
        acc_ang = np.array([
            (U2 + (self.Iy - self.Iz) * vel[4] * vel[5]) / self.Ix,
            (U3 + (self.Iz - self.Ix) * vel[3] * vel[5]) / self.Iy,
            (U4 + (self.Ix - self.Iy) * vel[3] * vel[4]) / self.Iz
        ])
        
        return np.concatenate([vel, acc_lin, acc_ang])
    
    def run_comparison(self):
        """
        Ejecuta la comparación completa entre ZN, PSO estándar y PSO con perturbación
        """
        # Combinaciones deseadas para pruebas
        desired_combinations = np.array([
            [1.0,  0.0,   0.0,    0.0],
            [1.5,  0.1,  -0.1,    0.0],
            [2.0, -0.2,   0.2,    0.0],
            [1.0,  0.0,   0.0,    np.pi/4],
            [0.5, -0.1,  -0.1,   -np.pi/6]
        ])
        
        results_summary = []
        
        for i, (z_des, phi_des, theta_des, psi_des) in enumerate(desired_combinations):
            print(f"\n{'='*50}")
            print(f"MANIOBRA {i+1}: z={z_des}, ϕ={phi_des}, θ={theta_des}, ψ={psi_des}")
            print(f"{'='*50}")
            
            # 1. Ziegler-Nichols
            zn_gains = self.ziegler_nichols_tuning(z_des, phi_des, theta_des, psi_des)
            zn_fitness, zn_metrics, t_zn, z_zn = self.evaluate_pid(
                zn_gains, z_des, phi_des, theta_des, psi_des, disturbance=False)
            
            # 2. PSO estándar (inicializado con ZN)
            pso_std_best, pso_std_metrics, conv_std, t_std, z_std = self.pso_optimization(
                z_des, phi_des, theta_des, psi_des, initial_gains=zn_gains, disturbance=False)
            
            # 3. PSO con perturbación (inicializado con PSO estándar)
            pso_dist_best, pso_dist_metrics, conv_dist, t_dist, z_dist = self.pso_optimization(
                z_des, phi_des, theta_des, psi_des, initial_gains=pso_std_best['position'], disturbance=True)
            
            # Almacenar resultados
            results_summary.append({
                'maneuver': i+1,
                'desired': [z_des, phi_des, theta_des, psi_des],
                'ZN': {'fitness': zn_fitness, 'metrics': zn_metrics, 'gains': zn_gains},
                'PSO_std': {'fitness': pso_std_best['fitness'], 'metrics': pso_std_metrics, 'gains': pso_std_best['position']},
                'PSO_dist': {'fitness': pso_dist_best['fitness'], 'metrics': pso_dist_metrics, 'gains': pso_dist_best['position']},
                'trajectories': {
                    'ZN': (t_zn, z_zn),
                    'PSO_std': (t_std, z_std),
                    'PSO_dist': (t_dist, z_dist)
                }
            })
            
            # Generar gráfica comparativa para esta maniobra
            self.plot_maneuver_comparison(i+1, z_des, results_summary[-1])
        
        # Generar resumen final
        self.generate_summary_report(results_summary)
        
        return results_summary
    
    def plot_maneuver_comparison(self, maneuver_num, z_des, results):
        """
        Genera gráfica comparativa para una maniobra específica
        """
        plt.figure(figsize=(12, 8))
        
        # Configurar subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfica superior: Comparación de respuestas
        t_zn, z_zn = results['trajectories']['ZN']
        t_std, z_std = results['trajectories']['PSO_std']
        t_dist, z_dist = results['trajectories']['PSO_dist']
        
        ax1.plot(t_zn, z_zn, 'b-', linewidth=2, label='Ziegler-Nichols')
        ax1.plot(t_std, z_std, 'g-', linewidth=2, label='PSO Estándar')
        ax1.plot(t_dist, z_dist, 'r-', linewidth=2, label='PSO con Perturbación')
        ax1.axhline(y=z_des, color='k', linestyle='--', alpha=0.7, label='Valor deseado')
        
        # Marcar zona de perturbación si aplica
        ax1.axvspan(2, 7, alpha=0.2, color='gray', label='Zona de perturbación')
        
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Altura z (m)')
        ax1.set_title(f'Comparación de Controladores - Maniobra {maneuver_num}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica inferior: Comparación de métricas
        methods = ['ZN', 'PSO_std', 'PSO_dist']
        labels = ['Ziegler-Nichols', 'PSO Estándar', 'PSO con Perturbación']
        fitness_values = [results[m]['fitness'] for m in methods]
        settling_times = [results[m]['metrics']['t_settle'] for m in methods]
        overshoots = [results[m]['metrics']['overshoot'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        # Normalizar para mejor visualización
        fitness_norm = [f/min(fitness_values) for f in fitness_values]
        settle_norm = [s/min(settling_times) for s in settling_times]
        overshoot_norm = [o/min([x for x in overshoots if x > 0]) for o in overshoots]
        
        ax2.bar(x - width, fitness_norm, width, label='Fitness (normalizado)')
        ax2.bar(x, settle_norm, width, label='Tiempo establecimiento (normalizado)')
        ax2.bar(x + width, overshoot_norm, width, label='Sobreimpulso (normalizado)')
        
        ax2.set_xlabel('Método de Control')
        ax2.set_ylabel('Métricas Normalizadas')
        ax2.set_title('Comparación de Métricas de Desempeño')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'Maniobra_{maneuver_num}_Comparacion.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfica de maniobra {maneuver_num} guardada.")
    
    def generate_summary_report(self, results_summary):
        """
        Genera un reporte resumen de todas las comparaciones
        """
        print("\n" + "="*80)
        print("REPORTE FINAL DE COMPARACIÓN")
        print("="*80)
        
        # Crear DataFrame para resumen
        summary_data = []
        
        for result in results_summary:
            maneuver = result['maneuver']
            z_des, phi_des, theta_des, psi_des = result['desired']
            
            for method in ['ZN', 'PSO_std', 'PSO_dist']:
                data = result[method]
                summary_data.append({
                    'Maniobra': maneuver,
                    'Método': method,
                    'z_deseado': z_des,
                    'Fitness': data['fitness'],
                    'Tiempo_Establecimiento': data['metrics']['t_settle'],
                    'Sobreimpulso_%': data['metrics']['overshoot'],
                    'Tiempo_Levantamiento': data['metrics']['t_rise'],
                    'Error_Estado_Estacionario': data['metrics']['steady_error'],
                    'ITSE': data['metrics']['ITSE'],
                    'IAE': data['metrics']['IAE'],
                    'RMSE': data['metrics']['RMSE']
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Guardar en Excel
        df_summary.to_excel('Resumen_Comparacion_Controladores.xlsx', index=False)
        
        # Imprimir resumen por consola
        print("\nRESUMEN POR MÉTODO (promedio de todas las maniobras):")
        print("-" * 100)
        
        for method in ['ZN', 'PSO_std', 'PSO_dist']:
            method_data = df_summary[df_summary['Método'] == method]
            avg_fitness = method_data['Fitness'].mean()
            avg_settle = method_data['Tiempo_Establecimiento'].mean()
            avg_overshoot = method_data['Sobreimpulso_%'].mean()
            
            print(f"{method:12} | Fitness: {avg_fitness:.4f} | "
                  f"Tiempo Est: {avg_settle:.3f}s | Sobreimpulso: {avg_overshoot:.2f}%")
        
        print("\nArchivo 'Resumen_Comparacion_Controladores.xlsx' guardado.")
        print("Gráficas individuales de cada maniobra guardadas.")


# Ejecutar la comparación completa
if __name__ == "__main__":
    optimizer = QuadrotorPIDOptimizer()
    results = optimizer.run_comparison()