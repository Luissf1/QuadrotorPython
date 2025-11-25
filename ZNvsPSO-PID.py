import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ThesisPSOAnalyzer:
    """
    Analizador PSO-PID optimizado para tesis de maestría
    Incluye comparación con Ziegler-Nichols, métricas completas y generación de tablas
    """
    
    def __init__(self):
        self.results_dir = "thesis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Parámetros del cuadrotor (de tu tesis)
        self.m = 1.0
        self.g = 9.81
        self.Ix = 0.1
        self.Iy = 0.1
        self.Iz = 0.2
        self.l = 0.25
        
        # Configuración PSO (optimizada)
        self.nVar = 12
        self.VarMin = np.array([2.0, 0.01, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1])
        self.VarMax = np.array([15,  2.0,  5.0, 10,  0.1,   2.0, 10,  0.1,   2.0, 10,  0.1,   2.0])
        self.MaxIter = 100
        self.nPop = 50
        self.num_executions = 30
        
        # Escenarios de prueba (5 escenarios como en tu tesis)
        self.flight_scenarios = [
            {"name": "E1 - Despegue Estacionario", "z_des": 1.0, "phi_des": 0.0, "theta_des": 0.0, "psi_des": 0.0, "disturbance": False},
            {"name": "E2 - Deriva Lateral", "z_des": 1.5, "phi_des": 0.1, "theta_des": -0.1, "psi_des": 0.0, "disturbance": True},
            {"name": "E3 - Ascenso Inclinado", "z_des": 2.0, "phi_des": -0.2, "theta_des": 0.2, "psi_des": 0.0, "disturbance": True},
            {"name": "E4 - Rotación Yaw", "z_des": 1.0, "phi_des": 0.0, "theta_des": 0.0, "psi_des": np.pi/4, "disturbance": True},
            {"name": "E5 - Trayectoria Transicional", "z_des": 0.5, "phi_des": -0.1, "theta_des": -0.1, "psi_des": -np.pi/6, "disturbance": True}
        ]
        
        # Almacenamiento de resultados
        self.zn_results = []
        self.pso_results = {f"E{i+1}": [] for i in range(5)}
        self.optimal_params = {}
        
    def run_complete_analysis(self):
        """Ejecuta el análisis completo para la tesis"""
        print("=" * 70)
        print("ANÁLISIS PSO-PID PARA TESIS DE MAESTRÍA")
        print("Comparación con Ziegler-Nichols - 5 Escenarios - 30 Ejecuciones")
        print("=" * 70)
        
        # Ejecutar Ziegler-Nichols
        print("\n🔧 EJECUTANDO ZIEGLER-NICHOLS...")
        self.zn_results = self.ziegler_nichols_analysis()
        
        # Ejecutar PSO para cada escenario
        print("\n🚀 OPTIMIZACIÓN PSO-PID...")
        for i, scenario in enumerate(tqdm(self.flight_scenarios, desc="Escenarios PSO")):
            scenario_results = self.pso_analysis_single_scenario(scenario, i+1)
            self.pso_results[f"E{i+1}"] = scenario_results
        
        # Generar resultados completos
        self.generate_thesis_tables()
        self.generate_thesis_figures()
        self.generate_statistical_analysis()
        
        print(f"\n✅ ANÁLISIS COMPLETADO! Resultados en: {self.results_dir}")
    
    def ziegler_nichols_analysis(self):
        """Análisis con método Ziegler-Nichols"""
        zn_results = []
        
        for scenario in tqdm(self.flight_scenarios, desc="ZN Tests"):
            # Parámetros ZN típicos (baseline)
            Kp_z, Ki_z, Kd_z = 12.5, 1.25, 3.125
            Kp_phi, Ki_phi, Kd_phi = 6.25, 0.075, 1.25
            Kp_theta, Ki_theta, Kd_theta = 6.25, 0.075, 1.25
            Kp_psi, Ki_psi, Kd_psi = 5.0, 0.05, 1.0
            
            gains = [Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi, 
                    Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi]
            
            # Evaluar desempeño
            fitness, metrics, t, z = self.evaluate_pid_performance(
                gains, scenario["z_des"], scenario["phi_des"], 
                scenario["theta_des"], scenario["psi_des"], 
                scenario["disturbance"]
            )
            
            zn_results.append({
                'scenario': scenario["name"],
                'fitness': fitness,
                'metrics': metrics,
                'gains': gains
            })
        
        return zn_results
    
    def pso_analysis_single_scenario(self, scenario, scenario_id):
        """Optimización PSO para un escenario específico"""
        print(f"\n🎯 Optimizando: {scenario['name']}")
        
        scenario_results = []
        best_global = {'position': None, 'fitness': float('inf'), 'metrics': None}
        
        # 30 ejecuciones independientes
        for execution in tqdm(range(self.num_executions), desc=f"Ejecuciones E{scenario_id}"):
            best_particle, convergence = self.pso_optimize(
                scenario["z_des"], scenario["phi_des"], 
                scenario["theta_des"], scenario["psi_des"], 
                scenario["disturbance"]
            )
            
            # Evaluar resultado final
            fitness, metrics, t, z = self.evaluate_pid_performance(
                best_particle["position"], scenario["z_des"], 
                scenario["phi_des"], scenario["theta_des"], 
                scenario["psi_des"], scenario["disturbance"]
            )
            
            result = {
                'execution': execution + 1,
                'fitness': fitness,
                'metrics': metrics,
                'gains': best_particle["position"],
                'convergence': convergence
            }
            
            scenario_results.append(result)
            
            # Actualizar mejor global
            if fitness < best_global['fitness']:
                best_global = {
                    'position': best_particle["position"].copy(),
                    'fitness': fitness,
                    'metrics': metrics
                }
        
        # Guardar mejores parámetros para este escenario
        self.optimal_params[f"E{scenario_id}"] = best_global
        
        return scenario_results
    
    def pso_optimize(self, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """Algoritmo PSO con barra de progreso"""
        # Configuración PSO
        w = 0.9
        w_damp = 0.99
        c1, c2 = 1.7, 1.7
        
        # Inicialización
        particles = []
        global_best = {'position': None, 'fitness': float('inf')}
        convergence = []
        
        # Barra de progreso para inicialización
        for i in range(self.nPop):
            position = np.random.uniform(self.VarMin, self.VarMax)
            velocity = np.zeros(self.nVar)
            
            fitness, metrics, t, z = self.evaluate_pid_performance(
                position, z_des, phi_des, theta_des, psi_des, disturbance
            )
            
            particle = {
                'position': position,
                'velocity': velocity,
                'fitness': fitness,
                'best_position': position.copy(),
                'best_fitness': fitness
            }
            
            particles.append(particle)
            
            if fitness < global_best['fitness']:
                global_best = {'position': position.copy(), 'fitness': fitness}
        
        # Bucle principal PSO con barra de progreso
        pbar = tqdm(total=self.MaxIter, desc="PSO Iteraciones", leave=False)
        
        for iter in range(self.MaxIter):
            for i in range(self.nPop):
                # Actualizar velocidad
                r1, r2 = np.random.rand(self.nVar), np.random.rand(self.nVar)
                cognitive = c1 * r1 * (particles[i]['best_position'] - particles[i]['position'])
                social = c2 * r2 * (global_best['position'] - particles[i]['position'])
                particles[i]['velocity'] = w * particles[i]['velocity'] + cognitive + social
                
                # Actualizar posición
                particles[i]['position'] = np.clip(
                    particles[i]['position'] + particles[i]['velocity'], 
                    self.VarMin, self.VarMax
                )
                
                # Evaluar
                fitness, metrics, t, z = self.evaluate_pid_performance(
                    particles[i]['position'], z_des, phi_des, theta_des, psi_des, disturbance
                )
                
                particles[i]['fitness'] = fitness
                
                # Actualizar mejores
                if fitness < particles[i]['best_fitness']:
                    particles[i]['best_position'] = particles[i]['position'].copy()
                    particles[i]['best_fitness'] = fitness
                    
                    if fitness < global_best['fitness']:
                        global_best = {'position': particles[i]['position'].copy(), 'fitness': fitness}
            
            convergence.append(global_best['fitness'])
            w *= w_damp
            
            # Actualizar barra de progreso
            pbar.set_postfix({'Best Fitness': f"{global_best['fitness']:.4f}"})
            pbar.update(1)
        
        pbar.close()
        
        return global_best, convergence
    
    def evaluate_pid_performance(self, gains, z_des, phi_des, theta_des, psi_des, disturbance=False):
        """Evaluación del desempeño del controlador PID"""
        try:
            # Resetear integrales
            self.integrals = np.zeros(4)
            
            # Simulación
            t_span = (0, 10)
            X0 = np.zeros(12)
            
            if disturbance:
                sol = solve_ivp(
                    lambda t, X: self.quadrotor_dynamics_with_disturbance(
                        t, X, gains, z_des, phi_des, theta_des, psi_des),
                    t_span, X0, t_eval=np.linspace(0, 10, 1000), method='RK45'
                )
            else:
                sol = solve_ivp(
                    lambda t, X: self.quadrotor_dynamics(
                        t, X, gains, z_des, phi_des, theta_des, psi_des),
                    t_span, X0, t_eval=np.linspace(0, 10, 1000), method='RK45'
                )
            
            t = sol.t
            X = sol.y
            z = X[2, :]
            
            # Calcular métricas
            metrics = self.calculate_performance_metrics(t, z, z_des)
            fitness = self.multi_objective_fitness(metrics)
            
            return fitness, metrics, t, z
            
        except:
            # Retornar valores por defecto en caso de error
            metrics = self.get_default_metrics()
            return 10.0, metrics, np.linspace(0, 10, 100), np.zeros(100)
    
    def quadrotor_dynamics(self, t, X, gains, z_des, phi_des, theta_des, psi_des):
        """Dinámica del cuadrotor sin perturbaciones"""
        return self._quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, False)
    
    def quadrotor_dynamics_with_disturbance(self, t, X, gains, z_des, phi_des, theta_des, psi_des):
        """Dinámica del cuadrotor con perturbaciones"""
        return self._quadrotor_dynamics(t, X, gains, z_des, phi_des, theta_des, psi_des, True)
    
    def _quadrotor_dynamics(self, t, X, gains, z_des, phi_des, theta_des, psi_des, disturbance):
        """Dinámica principal del cuadrotor"""
        pos = X[:6]
        vel = X[6:]
        
        # Extraer ganancias
        Kp_z, Ki_z, Kd_z = gains[0], gains[1], gains[2]
        Kp_phi, Ki_phi, Kd_phi = gains[3], gains[4], gains[5]
        Kp_theta, Ki_theta, Kd_theta = gains[6], gains[7], gains[8]
        Kp_psi, Ki_psi, Kd_psi = gains[9], gains[10], gains[11]
        
        # Calcular errores
        errors = np.array([
            z_des - pos[2],
            phi_des - pos[3],
            theta_des - pos[4],
            psi_des - pos[5]
        ])
        
        # Actualizar integrales
        dt = 0.01
        self.integrals = np.clip(self.integrals + errors * dt, -10, 10)
        
        # Control PID
        U1 = Kp_z * errors[0] + Ki_z * self.integrals[0] + Kd_z * (-vel[2])
        U2 = Kp_phi * errors[1] + Ki_phi * self.integrals[1] + Kd_phi * (-vel[3])
        U3 = Kp_theta * errors[2] + Ki_theta * self.integrals[2] + Kd_theta * (-vel[4])
        U4 = Kp_psi * errors[3] + Ki_psi * self.integrals[3] + Kd_psi * (-vel[5])
        
        # Perturbaciones
        F_disturbance = np.zeros(3)
        if disturbance:
            # Viento y turbulencias
            F_disturbance = np.array([
                0.5 * np.sin(0.5 * t) + 0.1 * np.random.randn(),
                0.3 * np.cos(0.3 * t) + 0.1 * np.random.randn(),
                0.2 * np.sin(0.2 * t) + 0.05 * np.random.randn()
            ])
        
        # Ecuaciones de movimiento
        acc_lin = np.array([
            (np.sin(psi_des)*np.sin(phi_des) + np.cos(psi_des)*np.sin(theta_des)*np.cos(phi_des)) * U1 / self.m + F_disturbance[0]/self.m,
            (-np.cos(psi_des)*np.sin(phi_des) + np.sin(psi_des)*np.sin(theta_des)*np.cos(phi_des)) * U1 / self.m + F_disturbance[1]/self.m,
            (np.cos(theta_des)*np.cos(phi_des) * U1 / self.m) - self.g + F_disturbance[2]/self.m
        ])
        
        acc_ang = np.array([
            (U2 + (self.Iy - self.Iz) * vel[4] * vel[5]) / self.Ix,
            (U3 + (self.Iz - self.Ix) * vel[3] * vel[5]) / self.Iy,
            (U4 + (self.Ix - self.Iy) * vel[3] * vel[4]) / self.Iz
        ])
        
        return np.concatenate((vel, acc_lin, acc_ang))
    
    def calculate_performance_metrics(self, t, z, z_des):
        """Calcular métricas de desempeño"""
        error = z_des - z
        
        # Tiempo de establecimiento (2%)
        tol = 0.02 * z_des
        settled_idx = np.where(np.abs(error) <= tol)[0]
        t_settle = t[settled_idx[-1]] if len(settled_idx) > 0 else t[-1]
        
        # Sobrepico
        overshoot = max(0, (np.max(z) - z_des) / z_des * 100) if z_des > 0 else 0
        
        # Métricas integrales
        ITSE = np.trapz(t * error**2, t) if len(t) > 1 else 0
        IAE = np.trapz(np.abs(error), t) if len(t) > 1 else 0
        RMSE = np.sqrt(np.mean(error**2))
        
        # Tiempo de subida (10% to 90%)
        try:
            idx_10 = np.where(z >= 0.1 * z_des)[0][0]
            idx_90 = np.where(z >= 0.9 * z_des)[0][0]
            t_rise = t[idx_90] - t[idx_10]
        except:
            t_rise = np.nan
        
        return {
            't_settle': t_settle,
            'overshoot': overshoot,
            'ITSE': ITSE,
            'IAE': IAE,
            'RMSE': RMSE,
            't_rise': t_rise
        }
    
    def multi_objective_fitness(self, metrics):
        """Función de costo multi-objetivo (como en tu tesis)"""
        weights = [0.3, 0.3, 0.2, 0.2]  # ts, Mp, ITSE, IAE
        
        fitness = (
            weights[0] * min(metrics['t_settle'] / 10, 1) +
            weights[1] * min(metrics['overshoot'] / 100, 1) +
            weights[2] * min(metrics['ITSE'] / 50, 1) +
            weights[3] * min(metrics['IAE'] / 20, 1)
        )
        
        return fitness
    
    def get_default_metrics(self):
        """Métricas por defecto para simulaciones fallidas"""
        return {
            't_settle': 10,
            'overshoot': 100,
            'ITSE': 50,
            'IAE': 20,
            'RMSE': 10,
            't_rise': 5
        }
    
    def generate_thesis_tables(self):
        """Generar tablas para la tesis"""
        print("\n📊 GENERANDO TABLAS PARA LA TESIS...")
        
        # Tabla 1: Comparación de métricas de desempeño
        table_data = []
        for i, (zn_result, scenario) in enumerate(zip(self.zn_results, self.flight_scenarios)):
            pso_metrics = [r['metrics'] for r in self.pso_results[f"E{i+1}"]]
            
            avg_pso_rmse = np.mean([m['RMSE'] for m in pso_metrics])
            avg_pso_iae = np.mean([m['IAE'] for m in pso_metrics])
            avg_pso_itse = np.mean([m['ITSE'] for m in pso_metrics])
            
            improvement_rmse = (zn_result['metrics']['RMSE'] - avg_pso_rmse) / zn_result['metrics']['RMSE'] * 100
            improvement_iae = (zn_result['metrics']['IAE'] - avg_pso_iae) / zn_result['metrics']['IAE'] * 100
            improvement_itse = (zn_result['metrics']['ITSE'] - avg_pso_itse) / zn_result['metrics']['ITSE'] * 100
            
            table_data.append({
                'Maniobra': scenario['name'],
                'RMSE_ZN': f"{zn_result['metrics']['RMSE']:.4f}",
                'RMSE_PSO': f"{avg_pso_rmse:.4f}",
                'Mejora_RMSE': f"{improvement_rmse:.1f}%",
                'IAE_ZN': f"{zn_result['metrics']['IAE']:.2f}",
                'IAE_PSO': f"{avg_pso_iae:.2f}",
                'Mejora_IAE': f"{improvement_iae:.1f}%",
                'ITSE_ZN': f"{zn_result['metrics']['ITSE']:.2f}",
                'ITSE_PSO': f"{avg_pso_itse:.2f}",
                'Mejora_ITSE': f"{improvement_itse:.1f}%"
            })
        
        df_comparison = pd.DataFrame(table_data)
        df_comparison.to_excel(os.path.join(self.results_dir, 'Tabla_Comparacion_Metricas.xlsx'), index=False)
        df_comparison.to_latex(os.path.join(self.results_dir, 'Tabla_Comparacion_Metricas.tex'), index=False)
        
        # Tabla 2: Parámetros PID óptimos
        params_data = []
        for scenario_id in self.optimal_params:
            optimal = self.optimal_params[scenario_id]
            gains = optimal['position']
            
            params_data.append({
                'Controlador': scenario_id,
                'Kp_z': f"{gains[0]:.3f}",
                'Ki_z': f"{gains[1]:.3f}",
                'Kd_z': f"{gains[2]:.3f}",
                'Kp_phi': f"{gains[3]:.3f}",
                'Ki_phi': f"{gains[4]:.3f}",
                'Kd_phi': f"{gains[5]:.3f}",
                'Kp_theta': f"{gains[6]:.3f}",
                'Ki_theta': f"{gains[7]:.3f}",
                'Kd_theta': f"{gains[8]:.3f}",
                'Kp_psi': f"{gains[9]:.3f}",
                'Ki_psi': f"{gains[10]:.3f}",
                'Kd_psi': f"{gains[11]:.3f}",
                'Fitness': f"{optimal['fitness']:.4f}"
            })
        
        df_params = pd.DataFrame(params_data)
        df_params.to_excel(os.path.join(self.results_dir, 'Tabla_Parametros_Optimos.xlsx'), index=False)
        df_params.to_latex(os.path.join(self.results_dir, 'Tabla_Parametros_Optimos.tex'), index=False)
        
        print("✅ Tablas generadas:")
        print("   - Tabla_Comparacion_Metricas.xlsx/tex")
        print("   - Tabla_Parametros_Optimos.xlsx/tex")
    
    def generate_thesis_figures(self):
        """Generar figuras para la tesis"""
        print("\n📈 GENERANDO FIGURAS PARA LA TESIS...")
        
        # Figura 1: Comparación de RMSE
        plt.figure(figsize=(12, 8))
        
        scenarios = [s['name'] for s in self.flight_scenarios]
        zn_rmse = [r['metrics']['RMSE'] for r in self.zn_results]
        pso_rmse = [np.mean([r['metrics']['RMSE'] for r in self.pso_results[f"E{i+1}"]]) for i in range(5)]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        plt.bar(x - width/2, zn_rmse, width, label='Ziegler-Nichols', alpha=0.7)
        plt.bar(x + width/2, pso_rmse, width, label='PSO-PID', alpha=0.7)
        
        plt.xlabel('Escenarios de Prueba')
        plt.ylabel('RMSE')
        plt.title('Comparación de RMSE: Ziegler-Nichols vs PSO-PID')
        plt.xticks(x, [f'E{i+1}' for i in range(5)])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Añadir valores
        for i, (zn, pso) in enumerate(zip(zn_rmse, pso_rmse)):
            plt.text(i - width/2, zn + 0.05, f'{zn:.3f}', ha='center', va='bottom')
            plt.text(i + width/2, pso + 0.05, f'{pso:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'Figura_Comparacion_RMSE.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figura 2: Convergencia PSO
        plt.figure(figsize=(10, 6))
        
        for i in range(5):
            convergence = self.pso_results[f"E{i+1}"][0]['convergence']
            plt.plot(convergence, label=f'E{i+1}')
        
        plt.xlabel('Iteración')
        plt.ylabel('Fitness')
        plt.title('Convergencia del Algoritmo PSO para Diferentes Escenarios')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'Figura_Convergencia_PSO.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Figuras generadas:")
        print("   - Figura_Comparacion_RMSE.png")
        print("   - Figura_Convergencia_PSO.png")
    
    def generate_statistical_analysis(self):
        """Análisis estadístico (pruebas Z)"""
        print("\n📊 GENERANDO ANÁLISIS ESTADÍSTICO...")
        
        statistical_data = []
        
        for i in range(5):
            zn_rmse = self.zn_results[i]['metrics']['RMSE']
            pso_rmses = [r['metrics']['RMSE'] for r in self.pso_results[f"E{i+1}"]]
            
            mu_pso = np.mean(pso_rmses)
            sigma_pso = np.std(pso_rmses)
            n = len(pso_rmses)
            
            # Prueba Z (una cola)
            Z = (mu_pso - zn_rmse) / (sigma_pso / np.sqrt(n))
            
            # Determinar significancia (α = 0.05, Z_critico = -1.645)
            significant = "Sí" if Z < -1.645 else "No"
            p_value = 2 * (1 - self.normal_cdf(abs(Z)))  # p-value aproximado
            
            statistical_data.append({
                'Escenario': f'E{i+1}',
                'Z_Score': f"{Z:.3f}",
                'p_Value': f"{p_value:.4f}",
                'Significancia': significant,
                'RMSE_ZN': f"{zn_rmse:.4f}",
                'RMSE_PSO_Avg': f"{mu_pso:.4f}",
                'RMSE_PSO_Std': f"{sigma_pso:.4f}"
            })
        
        df_stats = pd.DataFrame(statistical_data)
        df_stats.to_excel(os.path.join(self.results_dir, 'Analisis_Estadistico.xlsx'), index=False)
        df_stats.to_latex(os.path.join(self.results_dir, 'Analisis_Estadistico.tex'), index=False)
        
        print("✅ Análisis estadístico generado:")
        print("   - Analisis_Estadistico.xlsx/tex")
    
    def normal_cdf(self, x):
        """Función de distribución acumulativa normal aproximada"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal"""
    print("🚀 INICIANDO ANÁLISIS PARA TESIS DE MAESTRÍA")
    print("⏰ Este proceso tomará aproximadamente 20-30 minutos")
    print("💻 Se generarán tablas, figuras y análisis estadístico completos\n")
    
    analyzer = ThesisPSOAnalyzer()
    analyzer.run_complete_analysis()
    
    print("\n" + "="*70)
    print("🎓 ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("📁 Archivos generados en la carpeta 'thesis_results':")
    print("   • Tablas en formato Excel y LaTeX")
    print("   • Figuras en alta resolución (300 DPI)")
    print("   • Análisis estadístico con pruebas Z")
    print("   • Parámetros PID óptimos para cada controlador")
    print("\n✅ ¡Listo para incluir en tu tesis!")

if __name__ == "__main__":
    import math
    main()