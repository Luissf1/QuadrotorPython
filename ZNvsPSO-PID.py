import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os
from datetime import datetime

# =============================================================================
# MÓDULO ZIEGLER-NICHOLS (CORREGIDO)
# =============================================================================

class ZNController:
    """Controlador ZN con estado interno propio"""
    def __init__(self):
        self.integral_z = 0
        self.integral_phi = 0
        self.integral_theta = 0
        self.integral_psi = 0
        self.prev_error_z = 0
        self.prev_error_phi = 0
        self.prev_error_theta = 0
        self.prev_error_psi = 0
        self.last_time = 0

def ziegler_nichols_tuning_corrected(flight_conditions):
    """Versión corregida de Ziegler-Nichols"""
    m, g, Ix, Iy, Iz = 1.0, 9.81, 0.1, 0.1, 0.2
    RMSE_results = []
    
    for i, (z_des, phi_des, theta_des, psi_des) in enumerate(flight_conditions):
        print(f'--- ZN Flight {i+1}: z={z_des}, phi={phi_des:.1f}, theta={theta_des:.1f} ---')
        
        # Gains conservadoras - CORREGIDAS
        Kp_z, Ki_z, Kd_z = 12.0, 1.0, 5.0      # Más agresivo para altura
        Kp_phi, Ki_phi, Kd_phi = 6.0, 0.5, 1.0  # Más suave para ángulos
        Kp_theta, Ki_theta, Kd_theta = 6.0, 0.5, 1.0
        Kp_psi, Ki_psi, Kd_psi = 4.0, 0.2, 0.8
        
        try:
            # Estado inicial para CADA simulación
            X0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            controller = ZNController()
            
            t_span, t_eval = (0, 10), np.linspace(0, 10, 1000)
            
            sol = solve_ivp(
                lambda t, X: quadrotor_dynamics_zn_corrected(
                    t, X, m, g, Ix, Iy, Iz, Kp_z, Ki_z, Kd_z, 
                    Kp_phi, Ki_phi, Kd_phi, Kp_theta, Ki_theta, Kd_theta,
                    Kp_psi, Ki_psi, Kd_psi, z_des, phi_des, theta_des, 
                    psi_des, controller),
                t_span, X0, t_eval=t_eval, method='RK45', rtol=1e-6
            )
            
            if sol.success and len(sol.y) > 0:
                z = sol.y[2]
                # Filtrar valores válidos
                valid_indices = ~np.isnan(z)
                if np.any(valid_indices):
                    z_valid = z[valid_indices]
                    RMSE = np.sqrt(np.mean((z_des - z_valid)**2))
                    RMSE_results.append(RMSE)
                    print(f'  RMSE ZN: {RMSE:.4f}')
                else:
                    RMSE_results.append(np.inf)
                    print('  No valid z data!')
            else:
                RMSE_results.append(np.inf)
                print('  Simulation failed!')
                
        except Exception as e:
            RMSE_results.append(np.inf)
            print(f'  Error: {e}')
    
    print('\n=== RESULTADOS ZIEGLER-NICHOLS CORREGIDOS ===')
    for i, (cond, rmse) in enumerate(zip(flight_conditions, RMSE_results)):
        print(f'Test {i+1}: RMSE = {rmse:.4f}')
    
    return RMSE_results

def quadrotor_dynamics_zn_corrected(t, X, m, g, Ix, Iy, Iz,
                                  Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                                  Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                                  z_des, phi_des, theta_des, psi_des, controller):
    """Dinámica corregida para ZN"""
    
    # Extraer estados actuales
    current_z, current_phi, current_theta, current_psi = X[2], X[3], X[4], X[5]
    vel_z, vel_phi, vel_theta, vel_psi = X[8], X[9], X[10], X[11]
    
    # Calcular dt
    dt = t - controller.last_time if t > controller.last_time else 0.01
    controller.last_time = t
    
    # Errores
    error_z = z_des - current_z
    error_phi = phi_des - current_phi
    error_theta = theta_des - current_theta
    error_psi = psi_des - current_psi
    
    # Derivadas (usando velocidades reales en lugar de derivadas numéricas)
    derror_z = -vel_z  # La derivada del error es -velocidad
    derror_phi = -vel_phi
    derror_theta = -vel_theta
    derror_psi = -vel_psi
    
    # Integrales con anti-windup y límites más conservadores
    max_int = 2.0
    controller.integral_z = np.clip(controller.integral_z + error_z * dt, -max_int, max_int)
    controller.integral_phi = np.clip(controller.integral_phi + error_phi * dt, -max_int, max_int)
    controller.integral_theta = np.clip(controller.integral_theta + error_theta * dt, -max_int, max_int)
    controller.integral_psi = np.clip(controller.integral_psi + error_psi * dt, -max_int, max_int)
    
    # Control PID con límites más realistas
    # Fuerza vertical - asegurar que sea positiva y suficiente para contrarrestar gravedad
    U1_base = Kp_z * error_z + Ki_z * controller.integral_z + Kd_z * derror_z
    U1 = max(0.7 * m * g, min(U1_base, 2.0 * m * g))  # Límites más realistas
    
    # Momentos - más conservadores
    U2 = np.clip(Kp_phi * error_phi + Ki_phi * controller.integral_phi + Kd_phi * derror_phi, -1.0, 1.0)
    U3 = np.clip(Kp_theta * error_theta + Ki_theta * controller.integral_theta + Kd_theta * derror_theta, -1.0, 1.0)
    U4 = np.clip(Kp_psi * error_psi + Ki_psi * controller.integral_psi + Kd_psi * derror_psi, -0.5, 0.5)
    
    # Dinámica simplificada pero más estable
    acc_z = (U1 / m) - g
    acc_phi, acc_theta, acc_psi = U2 / Ix, U3 / Iy, U4 / Iz
    
    # Pequeña amortiguación para estabilidad
    damping = 0.1
    acc_z -= damping * vel_z
    acc_phi -= damping * vel_phi
    acc_theta -= damping * vel_theta
    acc_psi -= damping * vel_psi
    
    # Sin movimiento lateral forzado
    acc_x, acc_y = 0, 0
    
    dXdt = np.concatenate([X[6:], [acc_x, acc_y, acc_z, acc_phi, acc_theta, acc_psi]])
    return dXdt

# =============================================================================
# MÓDULO PSO-PID CORREGIDO
# =============================================================================

class PSODynamics:
    """Dinámica encapsulada para evaluaciones PSO"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.integrals = np.zeros(4)  # z, phi, theta, psi
        self.last_time = 0
        
    def compute(self, t, X, gains, z_des, phi_des, theta_des, psi_des):
        """Dinámica del quadrotor para evaluación PSO"""
        if t == 0:
            self.reset()
            
        m, g, Ix, Iy, Iz = 1.0, 9.81, 0.1, 0.1, 0.2
        pos = X[:6]
        vel = X[6:]
        
        # Extraer ganancias
        Kp_z, Ki_z, Kd_z = gains[0], gains[1], gains[2]
        Kp_phi, Ki_phi, Kd_phi = gains[3], gains[4], gains[5]
        Kp_theta, Ki_theta, Kd_theta = gains[6], gains[7], gains[8]
        Kp_psi, Ki_psi, Kd_psi = gains[9], gains[10], gains[11]
        
        # Calcular dt
        dt = t - self.last_time if t > self.last_time else 0.01
        self.last_time = t
        
        # Errores
        errors = np.array([
            z_des - pos[2],
            phi_des - pos[3], 
            theta_des - pos[4],
            psi_des - pos[5]
        ])
        
        # Actualizar integrales con anti-windup
        max_int = 5.0
        self.integrals = np.clip(self.integrals + errors * dt, -max_int, max_int)
        
        # Control PID
        U1 = Kp_z * errors[0] + Ki_z * self.integrals[0] + Kd_z * (-vel[2])
        U2 = Kp_phi * errors[1] + Ki_phi * self.integrals[1] + Kd_phi * (-vel[3])
        U3 = Kp_theta * errors[2] + Ki_theta * self.integrals[2] + Kd_theta * (-vel[4])
        U4 = Kp_psi * errors[3] + Ki_psi * self.integrals[3] + Kd_psi * (-vel[5])
        
        # Límites de control
        U1 = max(0.5 * m * g, min(U1, 3.0 * m * g))
        U2 = np.clip(U2, -2.0, 2.0)
        U3 = np.clip(U3, -2.0, 2.0)
        U4 = np.clip(U4, -1.0, 1.0)
        
        # Dinámica del quadrotor
        acc_lin = np.array([
            (np.cos(pos[3]) * np.sin(pos[4]) * np.cos(pos[5]) + np.sin(pos[3]) * np.sin(pos[5])) * U1 / m,
            (np.cos(pos[3]) * np.sin(pos[4]) * np.sin(pos[5]) - np.sin(pos[3]) * np.cos(pos[5])) * U1 / m,
            (np.cos(pos[3]) * np.cos(pos[4]) * U1 / m) - g
        ])
        
        acc_ang = np.array([
            (U2 + (Iy - Iz) * vel[4] * vel[5]) / Ix,
            (U3 + (Iz - Ix) * vel[3] * vel[5]) / Iy,
            (U4 + (Ix - Iy) * vel[3] * vel[4]) / Iz
        ])
        
        # Amortiguamiento para estabilidad
        damping = 0.05
        acc_lin[2] -= damping * vel[2]  # Amortiguamiento en Z
        acc_ang -= damping * vel[3:]    # Amortiguamiento angular
        
        dXdt = np.concatenate((vel, acc_lin, acc_ang))
        return dXdt

def calculate_settling_time(t, response, setpoint, tolerance=0.02):
    """Calcular tiempo de establecimiento"""
    error = np.abs(response - setpoint)
    settled_indices = np.where(error <= tolerance * setpoint)[0]
    
    if len(settled_indices) > 0:
        # Encontrar el último punto que sale de la tolerancia
        for i in range(len(settled_indices)-1, 0, -1):
            if settled_indices[i] - settled_indices[i-1] > 1:
                return t[settled_indices[i]]
        return t[settled_indices[-1]]
    return t[-1]  # Nunca se establece

def evaluate_pid_corrected(gains, z_des, phi_des, theta_des, psi_des):
    """Función de evaluación corregida para PSO"""
    
    dynamics = PSODynamics()
    
    try:
        # Simulación más corta para eficiencia (puedes ajustar)
        t_span = (0, 8)  # Reducido de 10 a 8 segundos
        t_eval = np.linspace(0, 8, 400)  # Menos puntos
        
        sol = solve_ivp(
            lambda t, X: dynamics.compute(t, X, gains, z_des, phi_des, theta_des, psi_des),
            t_span, np.zeros(12), t_eval=t_eval, method='RK45', rtol=1e-6
        )
        
        if not sol.success:
            return 2.0, {}  # Penalización moderada por fallo de integración
        
        t, X = sol.t, sol.y
        z = X[2, :]
        
        # Verificar datos válidos
        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
            return 2.5, {}
        
        error_z = z_des - z
        
        # Métricas robustas
        metrics = {
            'RMSE': np.sqrt(np.mean(error_z**2)),
            'IAE': np.trapz(np.abs(error_z), t),
            'ITSE': np.trapz(t * error_z**2, t),
            'max_overshoot': max(0, (np.max(z) - z_des) / z_des * 100) if z_des > 0 else 0,
            'settling_time': calculate_settling_time(t, z, z_des),
            'steady_state_error': np.mean(np.abs(error_z[-50:])) if len(error_z) > 50 else np.mean(np.abs(error_z))
        }
        
        # FUNCIÓN DE FITNESS MEJORADA (sin saturación)
        weights = {
            'RMSE': 0.30,    # Mayor peso a RMSE
            'IAE': 0.20, 
            'ITSE': 0.20,
            'max_overshoot': 0.15,
            'settling_time': 0.10,
            'steady_state_error': 0.05
        }
        
        # Normalización con función suave (sin min(..., 1))
        fitness = (
            weights['RMSE'] * (1 - np.exp(-metrics['RMSE']/1.0)) +
            weights['IAE'] * (1 - np.exp(-metrics['IAE']/10.0)) +
            weights['ITSE'] * (1 - np.exp(-metrics['ITSE']/20.0)) +
            weights['max_overshoot'] * (1 - np.exp(-metrics['max_overshoot']/30.0)) +
            weights['settling_time'] * (1 - np.exp(-metrics['settling_time']/5.0)) +
            weights['steady_state_error'] * (1 - np.exp(-metrics['steady_state_error']/0.3))
        )
        
        # Penalizaciones adicionales por comportamiento peligroso
        if metrics['max_overshoot'] > 50:  # Overshoot excesivo
            fitness += 0.3
        if metrics['settling_time'] > 6:   # Muy lento
            fitness += 0.2
        if metrics['steady_state_error'] > 0.1:  # Error grande en estado estacionario
            fitness += 0.2
            
        return fitness, metrics
        
    except Exception as e:
        # Penalización inteligente basada en tipo de error
        error_str = str(e).lower()
        if "overflow" in error_str or "inf" in error_str:
            return 3.0, {}  # Muy inestable
        elif "max-iter" in error_str:
            return 2.2, {}  # Muy lento
        else:
            return 2.5, {}  # Error genérico

def optimize_pid_with_pso_corrected(z_des, phi_des, theta_des, psi_des):
    """
    PSO optimization corregido con criterio de parada mejorado
    """
    # PARÁMETROS PSO OPTIMIZADOS
    nVar = 12
    MaxIter = 80  # Reducido por criterio de parada temprana
    nPop = 25     # Población más pequeña para eficiencia
    
    # Rangos de búsqueda optimizados
    VarMin = np.array([
        # Z control: Kp, Ki, Kd
        4.0, 0.05, 0.5,   
        # Phi control  
        0.5, 0.005, 0.05,  
        # Theta control
        0.5, 0.005, 0.05,  
        # Psi control
        0.3, 0.002, 0.03
    ])
    
    VarMax = np.array([
        25.0, 2.5, 10.0,   # Z
        12.0, 0.8, 3.0,    # Phi  
        12.0, 0.8, 3.0,    # Theta
        8.0, 0.3, 1.5      # Psi
    ])
    
    # Inicialización
    particles = []
    global_best = {'position': None, 'fitness': float('inf')}
    convergence_data = []
    
    # Estrategia de inicialización estratificada
    for i in range(nPop):
        if i < nPop//3:
            # Conservador: valores bajos
            position = VarMin + 0.2 * (VarMax - VarMin) * np.random.rand(nVar)
        elif i < 2*nPop//3:
            # Moderado: valores medios
            position = VarMin + 0.5 * (VarMax - VarMin) * np.random.rand(nVar)
        else:
            # Agresivo: valores altos
            position = VarMin + 0.8 * (VarMax - VarMin) * np.random.rand(nVar)
        
        particle = {
            'position': position,
            'velocity': np.zeros(nVar),
            'fitness': float('inf'),
            'best': {'position': position.copy(), 'fitness': float('inf')}
        }
        
        particle['fitness'], _ = evaluate_pid_corrected(
            particle['position'], z_des, phi_des, theta_des, psi_des)
        
        particle['best']['fitness'] = particle['fitness']
        
        if particle['fitness'] < global_best['fitness']:
            global_best = particle['best'].copy()
        
        particles.append(particle)
    
    # Parámetros adaptativos
    w = 0.9     # Inercia inicial alta
    w_damp = 0.98
    c1_initial, c2_initial = 2.0, 2.0
    
    # Variables para criterio de parada temprana
    no_improvement_count = 0
    last_best_fitness = global_best['fitness']
    convergence_threshold = 1e-4
    max_no_improvement = 15  # Máximo de iteraciones sin mejora
    
    # Bucle de optimización con criterio de parada temprana
    for iter in range(MaxIter):
        # Coeficientes adaptativos
        progress = iter / MaxIter
        c1 = c1_initial * (1 - progress)  # Menor exploración
        c2 = c2_initial * progress        # Mayor explotación
        
        for i in range(nPop):
            r1, r2 = np.random.rand(nVar), np.random.rand(nVar)
            
            # Actualización de velocidad
            inertia = w * particles[i]['velocity']
            cognitive = c1 * r1 * (particles[i]['best']['position'] - particles[i]['position'])
            social = c2 * r2 * (global_best['position'] - particles[i]['position'])
            
            particles[i]['velocity'] = inertia + cognitive + social
            
            # Límite de velocidad
            max_velocity = 0.15 * (VarMax - VarMin)
            particles[i]['velocity'] = np.clip(particles[i]['velocity'], -max_velocity, max_velocity)
            
            # Actualizar posición
            particles[i]['position'] = np.clip(
                particles[i]['position'] + particles[i]['velocity'], VarMin, VarMax)
            
            # Evaluar
            fitness, metrics = evaluate_pid_corrected(
                particles[i]['position'], z_des, phi_des, theta_des, psi_des)
            
            particles[i]['fitness'] = fitness
            
            # Actualizar mejores
            if fitness < particles[i]['best']['fitness']:
                particles[i]['best']['position'] = particles[i]['position'].copy()
                particles[i]['best']['fitness'] = fitness
                
                if fitness < global_best['fitness']:
                    global_best = particles[i]['best'].copy()
                    no_improvement_count = 0  # Resetear contador
        
        # Actualizar inercia
        w = max(w * w_damp, 0.4)
        
        # Verificar convergencia
        convergence_data.append(global_best['fitness'])
        
        # Criterio de parada temprana
        fitness_improvement = abs(last_best_fitness - global_best['fitness'])
        if fitness_improvement < convergence_threshold:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            
        last_best_fitness = global_best['fitness']
        
        # Log de progreso
        if iter % 10 == 0:
            print(f'    Iter {iter}: Fitness = {global_best["fitness"]:.4f}, NoImprove = {no_improvement_count}')
        
        # Parar si no hay mejora significativa
        if no_improvement_count >= max_no_improvement:
            print(f'    ⏹️  Parada temprana en iteración {iter} (sin mejora por {max_no_improvement} iteraciones)')
            break
    
    print(f'    ✅ PSO completado: {iter+1} iteraciones, Fitness final: {global_best["fitness"]:.4f}')
    return global_best, convergence_data

def pso_pid_multiple_tests_corrected(flight_conditions, movimientos):
    """PSO-PID corregido con nuevo criterio de parada"""
    resultados_completos = []
    
    for i, (z_des, phi_des, theta_des, psi_des) in enumerate(flight_conditions):
        print(f'\n--- PSO Test {i+1}: {movimientos[i]} ---')
        print("Ejecutando PSO corregido (30 ejecuciones)...")
        
        rmse_values = []
        todos_resultados = []
        
        for test in range(30):
            global_best, convergence = optimize_pid_with_pso_corrected(
                z_des, phi_des, theta_des, psi_des)
            
            # Evaluar final con métricas completas
            fitness, metrics = evaluate_pid_corrected(
                global_best['position'], z_des, phi_des, theta_des, psi_des)
            
            rmse_values.append(metrics.get('RMSE', 10.0))  # Default alto si falla
            todos_resultados.append({
                'fitness': global_best['fitness'],
                'metrics': metrics,
                'gains': global_best['position'],
                'convergence': convergence
            })
            
            if (test + 1) % 5 == 0:
                print(f'    Completadas {test + 1}/30 ejecuciones...')
        
        sigma_pso = np.std(rmse_values)
        resultados_completos.append({
            'movimiento': movimientos[i],
            'mu_PSO': np.mean(rmse_values),
            'sigma_PSO': sigma_pso,
            'RMSE_values': rmse_values,
            'todos_resultados': todos_resultados
        })
        
        print(f'  PSO completado: mu_PSO: {np.mean(rmse_values):.4f}, sigma_PSO: {sigma_pso:.4f}')
    
    return resultados_completos

# =============================================================================
# MÓDULO ESTADÍSTICAS Y COMPARACIÓN (IGUAL QUE ANTES)
# =============================================================================

def calcular_estadisticas_z(RMSE_ZN, resultados_pso, movimientos):
    """Calcular estadísticas Z y generar tabla completa"""
    tabla_data = []
    
    for i, movimiento in enumerate(movimientos):
        rmse_zn = RMSE_ZN[i]
        mu_pso = resultados_pso[i]['mu_PSO']
        sigma_pso = resultados_pso[i]['sigma_PSO']
        n = 30  # ejecuciones PSO
        
        # Calcular Z según fórmula del director
        Z = (mu_pso - rmse_zn) / (sigma_pso / np.sqrt(n))
        
        # Determinar conclusión
        conclusion = "Significativa" if Z < -1.645 else "No significativa"
        
        tabla_data.append({
            'Movimiento': movimiento,
            'Z': f"{Z:.2f}",
            'RMSEzN': f"{rmse_zn:.4f}",
            'mu_PSO': f"{mu_pso:.4f}",
            'sigma_PSO': f"{sigma_pso:.4f}",
            'Conclusion': conclusion
        })
    
    return pd.DataFrame(tabla_data)

def generar_graficas_comparativas(RMSE_ZN, resultados_pso, movimientos):
    """Generar gráficas comparativas para el reporte"""
    mu_pso = [r['mu_PSO'] for r in resultados_pso]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfica 1: Comparación RMSE
    x_pos = np.arange(len(movimientos))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, RMSE_ZN, width, label='Ziegler-Nichols', 
                   color='red', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, mu_pso, width, label='PSO Optimizado', 
                   color='blue', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Escenarios de Prueba', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax1.set_title('COMPARACION DE RENDIMIENTO: ZN vs PSO', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Test {i+1}' for i in range(len(movimientos))])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, (zn, pso) in enumerate(zip(RMSE_ZN, mu_pso)):
        ax1.text(i - width/2, zn + 0.02, f'{zn:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
        ax1.text(i + width/2, pso + 0.02, f'{pso:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9)
    
    # Gráfica 2: Mejora porcentual
    mejoras = [(zn - pso)/zn * 100 for zn, pso in zip(RMSE_ZN, mu_pso)]
    colors = ['green' if mejora > 0 else 'red' for mejora in mejoras]
    
    bars3 = ax2.bar(x_pos, mejoras, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Escenarios de Prueba', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mejora (%)', fontsize=12, fontweight='bold')
    ax2.set_title('MEJORA PORCENTUAL DEL PSO SOBRE ZN', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Test {i+1}' for i in range(len(movimientos))])
    ax2.grid(True, alpha=0.3)
    
    # Añadir valores de mejora
    for i, mejora in enumerate(mejoras):
        ax2.text(i, mejora + (1 if mejora > 0 else -3), f'{mejora:.1f}%', 
                ha='center', va='bottom' if mejora > 0 else 'top', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comparacion_rmse_corregido.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# FUNCIÓN PRINCIPAL CORREGIDA
# =============================================================================

def analisis_comparativo_corregido():
    """
    Análisis comparativo completo con código corregido
    """
    print("=== ANALISIS COMPARATIVO CORREGIDO: ZN vs PSO-PID ===")
    
    # Configuración común
    flight_conditions = np.array([
        [1.0,  0.0,   0.0,    0.0],
        [1.5,  0.1,  -0.1,    0.0], 
        [2.0, -0.2,   0.2,    0.0],
        [1.0,  0.0,   0.0,    np.pi/4],
        [0.5, -0.1,  -0.1,   -np.pi/6]
    ])
    
    movimientos = [
        "Despegar sin inclinacion",
        "Despegar con inclinacion roll y pitch", 
        "Despegar sin inclinacion y con giro yaw",
        "Despegue controlado por yaw",
        "Despegue transicional y cambio de altitud"
    ]
    
    # Paso 1: Ejecutar Ziegler-Nichols corregido
    print("\n" + "="*60)
    print("EJECUTANDO ZIEGLER-NICHOLS CORREGIDO...")
    print("="*60)
    RMSE_ZN = ziegler_nichols_tuning_corrected(flight_conditions)
    
    # Paso 2: Ejecutar PSO-PID corregido
    print("\n" + "="*60)
    print("EJECUTANDO PSO-PID CORREGIDO...")
    print("="*60)
    print("NOTA: Con criterio de parada temprana - más eficiente")
    resultados_pso = pso_pid_multiple_tests_corrected(flight_conditions, movimientos)
    
    # Paso 3: Calcular estadísticas Z
    print("\n" + "="*60)
    print("CALCULANDO ESTADISTICAS Z...")
    print("="*60)
    tabla_completa = calcular_estadisticas_z(RMSE_ZN, resultados_pso, movimientos)
    
    # Paso 4: Generar reporte final
    print("\n" + "="*60)
    print("TABLA COMPLETA: PRUEBA Z TEST (CORREGIDA)")
    print("="*60)
    print(tabla_completa.to_string(index=False))
    
    # Paso 5: Generar gráficas comparativas
    generar_graficas_comparativas(RMSE_ZN, resultados_pso, movimientos)
    
    return tabla_completa, RMSE_ZN, resultados_pso

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Ejecutar análisis corregido
    try:
        tabla_final, RMSE_ZN, resultados_pso = analisis_comparativo_corregido()
        
        # Mostrar resumen final
        print("\n" + "="*60)
        print("ANALISIS CORREGIDO COMPLETADO EXITOSAMENTE")
        print("="*60)
        
        # Calcular estadísticas resumen
        mejoras = [(zn - pso['mu_PSO'])/zn * 100 for zn, pso in zip(RMSE_ZN, resultados_pso)]
        tests_significativos = sum(1 for r in tabla_final['Conclusion'] if r == 'Significativa')
        
        print(f"RESULTADOS OBTENIDOS (CORREGIDOS):")
        print(f"   - Tests significativos: {tests_significativos}/5")
        print(f"   - Mejora promedio: {np.mean(mejoras):.1f}%")
        print(f"   - Mejor mejora: {max(mejoras):.1f}%")
        print(f"   - Valor Z más extremo: {min([float(z) for z in tabla_final['Z']]):.2f}")
        
        print(f"EFICIENCIA COMPUTACIONAL:")
        print(f"   - PSO con criterio de parada temprana")
        print(f"   - Población reducida: 25 partículas")
        print(f"   - Iteraciones máximas: 80 (con parada temprana)")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()