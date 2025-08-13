import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
import os

def pso_pid_multiple_tests():
    """
    Main function to run multiple PSO-PID optimization tests for different desired positions.
    This is equivalent to the MATLAB pso_pid_multiple_tests() function.
    """
    # Desired combinations: [z, phi, theta, psi]
    desired_combinations = np.array([
        [1.0,  0.0,   0.0,    0.0],
        [1.5,  0.1,  -0.1,    0.0],
        [2.0, -0.2,   0.2,    0.0],
        [1.0,  0.0,   0.0,    np.pi/4],
        [0.5, -0.1,  -0.1,   -np.pi/6]
    ])

    for i in range(len(desired_combinations)):
        print(f'\n============ Test {i+1} ============')
        z_des = desired_combinations[i, 0]
        phi_des = desired_combinations[i, 1]
        theta_des = desired_combinations[i, 2]
        psi_des = desired_combinations[i, 3]

        excel_filename = f'Results_PSO_PID_Test_{i+1}.xlsx'
        fitness_figure_name = f'Convergence_Test_{i+1}.png'
        z_figure_name = f'ResponseZ_Test_{i+1}.png'

        pso_pid_optimization_with_metrics(
            z_des, phi_des, theta_des, psi_des,
            excel_filename, fitness_figure_name, z_figure_name
        )

def pso_pid_optimization_with_metrics(z_des, phi_des, theta_des, psi_des, 
                                     excel_filename, fitness_figure_name, z_figure_name):
    """
    Perform PSO-PID optimization with metrics collection and visualization.
    Equivalent to the MATLAB pso_pid_optimization_with_metrics() function.
    """
    num_tests = 30
    results = []
    best_fitness_over_time = []
    best_global_overall = {'fitness': float('inf')}
    best_results = []
    
    # Create directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    for test in range(num_tests):
        global_best, metrics, convergence_fitness, t_best, z_best = (
            optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des)
        )
        
        # Store convergence data
        best_fitness_over_time.append(convergence_fitness)
        
        # Create results dictionary for this test
        result = {
            'Test': test + 1,
            'Fitness': global_best['fitness'],
            'SettlingTime': metrics['t_settle'],
            'Overshoot': metrics['overshoot'],
            'RiseTime': metrics['t_rise'],
            'SteadyError': metrics['steady_error'],
            'ITSE': metrics['ITSE'],
            'IAE': metrics['IAE'],
            'RMSE': metrics['RMSE'],
            'Kp_z': global_best['position'][0],
            'Ki_z': global_best['position'][1],
            'Kd_z': global_best['position'][2],
            'Kp_phi': global_best['position'][3],
            'Ki_phi': global_best['position'][4],
            'Kd_phi': global_best['position'][5],
            'Kp_theta': global_best['position'][6],
            'Ki_theta': global_best['position'][7],
            'Kd_theta': global_best['position'][8],
            'Kp_psi': global_best['position'][9],
            'Ki_psi': global_best['position'][10],
            'Kd_psi': global_best['position'][11]
        }
        results.append(result)
        
        # Update overall best if this test is better
        if global_best['fitness'] < best_global_overall['fitness']:
            best_global_overall = global_best
            t_global_best = t_best
            z_global_best = z_best
        
        # Store best results for visualization
        best_results.append({
            't': t_best,
            'z': z_best,
            'label': f'Test {test + 1}'
        })
        
        # Print progress
        print(f"{test+1}\t{global_best['fitness']:.4f}\t{metrics['t_settle']:.4f}\t"
              f"{metrics['overshoot']:.2f}\t{metrics['t_rise']:.4f}\t"
              f"{metrics['steady_error']:.4f}\t{metrics['ITSE']:.4f}\t"
              f"{metrics['IAE']:.4f}\t{metrics['RMSE']:.4f}")
    
    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(os.path.join('results', excel_filename), index=False)
    print(f'\nFile saved: {excel_filename}')
    
    # Plot average convergence
    avg_convergence = np.mean(best_fitness_over_time, axis=0)
    plt.figure()
    plt.plot(avg_convergence, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Average PSO Convergence')
    plt.grid(True)
    plt.savefig(os.path.join('results', fitness_figure_name))
    plt.close()
    
    # Plot best Z response
    plt.figure()
    plt.plot(t_global_best, z_global_best, 'r', linewidth=2)
    plt.axhline(y=z_des, color='k', linestyle='--', label='Desired value')
    plt.xlabel('Time (s)')
    plt.ylabel('Height z (m)')
    plt.title('Height z Response for Best Solution')
    plt.grid(True)
    plt.savefig(os.path.join('results', z_figure_name))
    plt.close()
    
    # Plot top 5 trajectories
    sorted_indices = np.argsort([r['Fitness'] for r in results])
    top_results = [best_results[i] for i in sorted_indices[:5]]
    
    plt.figure()
    for res in top_results:
        plt.plot(res['t'], res['z'], linewidth=1.5, label=res['label'])
    plt.axhline(y=z_des, color='k', linestyle='--', label='Desired value')
    plt.xlabel('Time (s)')
    plt.ylabel('Height z (m)')
    plt.title('Comparison of Top 5 Trajectories')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join('results', z_figure_name.replace('.png', '_Top5.png')))
    plt.close()

def optimize_pid_with_pso_and_metrics(z_des, phi_des, theta_des, psi_des):
    """
    PSO optimization for PID controller parameters.
    Equivalent to the MATLAB optimize_pid_with_pso_and_metrics() function.
    """
    nVar = 12
    VarMin = np.array([2.0, 0.01, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1,  0.1, 0.001, 0.1])
    VarMax = np.array([15,  2.0,  5.0, 10,  0.1,   2.0, 10,  0.1,   2.0, 10,  0.1,   2.0])
    MaxIter = 100
    nPop = 50
    w = 0.7
    d = 0.97
    c1 = 1.7
    c2 = 1.7
    
    # Initialize particles
    particles = []
    global_best = {'position': None, 'fitness': float('inf')}
    B = np.zeros(MaxIter)
    t_best = []
    z_best = []
    metrics = None
    
    for i in range(nPop):
        particle = {
            'position': np.random.uniform(VarMin, VarMax),
            'velocity': np.zeros(nVar),
            'fitness': float('inf'),
            'best': {'position': None, 'fitness': float('inf')}
        }
        particle['fitness'], _, _, _ = evaluate_pid(particle['position'], z_des, phi_des, theta_des, psi_des)
        particle['best'] = {'position': particle['position'].copy(), 'fitness': particle['fitness']}
        
        if particle['fitness'] < global_best['fitness']:
            global_best = {'position': particle['position'].copy(), 'fitness': particle['fitness']}
        
        particles.append(particle)
    
    # PSO main loop
    for iter in range(MaxIter):
        for i in range(nPop):
            r1 = np.random.rand(nVar)
            r2 = np.random.rand(nVar)
            
            # Update velocity
            particles[i]['velocity'] = (w * particles[i]['velocity'] + 
                c1 * r1 * (particles[i]['best']['position'] - particles[i]['position']) + 
                c2 * r2 * (global_best['position'] - particles[i]['position']))
            
            # Update position
            particles[i]['position'] = np.clip(
                particles[i]['position'] + particles[i]['velocity'], VarMin, VarMax)
            
            # Evaluate new position
            fitness, temp_metrics, t, z = evaluate_pid(
                particles[i]['position'], z_des, phi_des, theta_des, psi_des)
            particles[i]['fitness'] = fitness
            
            # Update personal best
            if fitness < particles[i]['best']['fitness']:
                particles[i]['best']['position'] = particles[i]['position'].copy()
                particles[i]['best']['fitness'] = fitness
                
                # Update global best
                if fitness < global_best['fitness']:
                    global_best = {'position': particles[i]['position'].copy(), 'fitness': fitness}
                    metrics = temp_metrics
                    t_best = t
                    z_best = z
        
        # Update inertia weight
        w = max(w * d, 0.4)
        B[iter] = global_best['fitness']
    
    return global_best, metrics, B, t_best, z_best

def evaluate_pid(gains, z_des, phi_des, theta_des, psi_des):
    """
    Evaluate PID controller performance with given gains.
    Equivalent to the MATLAB evaluate_pid() function.
    """
    m = 1.0
    g = 9.81
    Ix = 0.1
    Iy = 0.1
    Iz = 0.2
    x0 = np.zeros(6)
    xdot0 = np.zeros(6)
    X0 = np.concatenate((x0, xdot0))
    t_span = (0, 10)
    
    # Extract PID gains
    Kp_z, Ki_z, Kd_z = gains[0], gains[1], gains[2]
    Kp_phi, Ki_phi, Kd_phi = gains[3], gains[4], gains[5]
    Kp_theta, Ki_theta, Kd_theta = gains[6], gains[7], gains[8]
    Kp_psi, Ki_psi, Kd_psi = gains[9], gains[10], gains[11]
    
    # Initialize metrics dictionary
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
        # Solve the ODE
        sol = solve_ivp(
            lambda t, X: quadrotor_dynamics(
                t, X, m, g, Ix, Iy, Iz,
                Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                z_des, phi_des, theta_des, psi_des),
            t_span, X0, t_eval=np.linspace(0, 10, 1000)
        )
        
        t = sol.t
        X = sol.y
        z = X[2, :]
        error_z = z_des - z
        
        # Calculate performance metrics
        tol = 0.02 * z_des
        idx_settle = np.where(np.abs(error_z) > tol)[0]
        metrics['t_settle'] = t[idx_settle[-1]] if len(idx_settle) > 0 else 0
        metrics['overshoot'] = max(0, (np.max(z) - z_des) / z_des * 100)
        
        rise_start = z_des * 0.1
        rise_end = z_des * 0.9
        try:
            t_rise_start = t[np.where(z >= rise_start)[0][0]]
            t_rise_end = t[np.where(z >= rise_end)[0][0]]
            metrics['t_rise'] = t_rise_end - t_rise_start
        except:
            metrics['t_rise'] = np.nan
        
        metrics['steady_error'] = np.mean(np.abs(error_z[int(0.9*len(error_z)):]))
        metrics['ITSE'] = np.trapz(t * error_z**2, t)
        metrics['IAE'] = np.trapz(np.abs(error_z), t)
        metrics['RMSE'] = np.sqrt(np.mean(error_z**2))
        
        # Calculate fitness function
        fitness = (0.3 * min(metrics['t_settle']/10, 1) + 
                   0.3 * min(metrics['overshoot']/100, 1) + 
                   0.2 * min(metrics['ITSE']/50, 1) + 
                   0.2 * min(metrics['IAE']/20, 1))
    
    except:
        # If simulation fails, return high fitness and default metrics
        fitness = 1000
        metrics = {
            't_settle': 100,
            'overshoot': 1000,
            't_rise': 10,
            'steady_error': 1,
            'ITSE': 50,
            'IAE': 20,
            'RMSE': 50
        }
        t = np.linspace(0, 10, 100)
        z = np.zeros_like(t)
    
    return fitness, metrics, t, z

def quadrotor_dynamics(t, X, m, g, Ix, Iy, Iz,
                      Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                      Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                      z_des, phi_des, theta_des, psi_des):
    """
    Quadrotor dynamics model with PID control.
    Equivalent to the MATLAB quadrotor_dynamics() function.
    """
    # Persistent variables for integral terms (using function attributes)
    if not hasattr(quadrotor_dynamics, 'iz'):
        quadrotor_dynamics.iz = 0
        quadrotor_dynamics.ip = 0
        quadrotor_dynamics.it = 0
        quadrotor_dynamics.ipsi = 0
    
    pos = X[:6]
    vel = X[6:]
    
    # Calculate errors
    err = np.array([
        z_des - pos[2],
        phi_des - pos[3],
        theta_des - pos[4],
        psi_des - pos[5]
    ])
    
    # Update integral terms with anti-windup
    max_int = 10
    quadrotor_dynamics.iz = np.clip(quadrotor_dynamics.iz + err[0], -max_int, max_int)
    quadrotor_dynamics.ip = np.clip(quadrotor_dynamics.ip + err[1], -max_int, max_int)
    quadrotor_dynamics.it = np.clip(quadrotor_dynamics.it + err[2], -max_int, max_int)
    quadrotor_dynamics.ipsi = np.clip(quadrotor_dynamics.ipsi + err[3], -max_int, max_int)
    
    # Calculate control inputs
    U1 = Kp_z * err[0] + Ki_z * quadrotor_dynamics.iz + Kd_z * (-vel[2])
    U2 = Kp_phi * err[1] + Ki_phi * quadrotor_dynamics.ip + Kd_phi * (-vel[3])
    U3 = Kp_theta * err[2] + Ki_theta * quadrotor_dynamics.it + Kd_theta * (-vel[4])
    U4 = Kp_psi * err[3] + Ki_psi * quadrotor_dynamics.ipsi + Kd_psi * (-vel[5])
    
    # Calculate linear and angular accelerations
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
    
    # Return state derivatives
    dXdt = np.concatenate((vel, acc_lin, acc_ang))
    return dXdt

if __name__ == "__main__":
    pso_pid_multiple_tests()