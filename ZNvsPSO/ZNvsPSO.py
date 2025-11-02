import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from pyswarm import pso
from math import cos, sin, pi
import warnings

# Suppress RunTimeWarning for division by zero or invalid values
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# 1. QUADROTOR PARAMETERS AND DYNAMICS
# ==============================================================================

# Parameters from Table 1 and Section 5.1
PARAMS = {
    'm': 1.0,         # Mass (kg)
    'g': 9.81,        # Gravity (m/s^2)
    'Ix': 0.1,        # Moment of inertia x (kg*m^2)
    'Iy': 0.1,        # Moment of inertia y (kg*m^2)
    'Iz': 0.2,        # Moment of inertia z (kg*m^2)
    'l': 0.25,        # Arm length (m) - Estimated/standard value
    'd': 0.01         # Drag factor - Estimated/standard value
}

# Initial state: [x, y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi]
X0 = np.zeros(12) 
TSPAN = (0, 10) # Simulation time (s)

# Control Integrator State
# This allows the integral error to persist across time steps in the ODE solver
class IntegralError:
    def __init__(self):
        self.iz, self.ip, self.it, self.ipsi = 0.0, 0.0, 0.0, 0.0
        self.max_int = 10.0 # Integral anti-windup limit

    def reset(self):
        self.iz, self.ip, self.it, self.ipsi = 0.0, 0.0, 0.0, 0.0
        
    def update(self, errors):
        # Update and apply anti-windup (saturation)
        self.iz = np.clip(self.iz + errors[0], -self.max_int, self.max_int)
        self.ip = np.clip(self.ip + errors[1], -self.max_int, self.max_int)
        self.it = np.clip(self.it + errors[2], -self.max_int, self.max_int)
        self.ipsi = np.clip(self.ipsi + errors[3], -self.max_int, self.max_int)
        return [self.iz, self.ip, self.it, self.ipsi]

# Global/Shared object for integral errors (used by ODE solver)
integral_error_state = IntegralError()

def quadrotor_dynamics(t, X, gains, des_refs):
    """
    Quadrotor ODE system based on simplified nonlinear model (Section 5.1).
    X is the 12-dimensional state vector.
    """
    m, g, Ix, Iy, Iz = PARAMS['m'], PARAMS['g'], PARAMS['Ix'], PARAMS['Iy'], PARAMS['Iz']
    
    # State extraction
    z, phi, theta, psi = X[2], X[3], X[4], X[5]
    dz, dphi, dtheta, dpsi = X[8], X[9], X[10], X[11]

    # Desired References
    z_des, phi_des, theta_des, psi_des = des_refs

    # PID Gains (12 elements)
    Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi, \
    Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi = gains

    # Error calculation
    errors = np.array([z_des - z, phi_des - phi, theta_des - theta, psi_des - psi])

    # Integral term update (uses integral_error_state class)
    integral_terms = integral_error_state.update(errors * (TSPAN[1] / 1000)) # Simple Euler integration
    
    # Control Inputs U1, U2, U3, U4 (Equation 5)
    U1 = Kp_z * errors[0] + Ki_z * integral_terms[0] + Kd_z * (-dz)
    U2 = Kp_phi * errors[1] + Ki_phi * integral_terms[1] + Kd_phi * (-dphi)
    U3 = Kp_theta * errors[2] + Ki_theta * integral_terms[2] + Kd_theta * (-dtheta)
    U4 = Kp_psi * errors[3] + Ki_psi * integral_terms[3] + Kd_psi * (-dpsi)

    # Dynamics (Equations from Section 5.1 - Simplified to single-axis)
    # The paper's dynamics are simplified for the purpose of PID tuning
    ddx = 0 # Not explicitly defined, typically assumed small or zero-mean
    ddy = 0 # Not explicitly defined
    ddz = (cos(phi) * cos(theta) / m) * U1 - g # Vertical acceleration
    
    ddphi = (U2 + (Iy - Iz) * dtheta * dpsi) / Ix # Roll acceleration
    ddtheta = (U3 + (Iz - Ix) * dphi * dpsi) / Iy # Pitch acceleration
    ddpsi = (U4 + (Ix - Iy) * dphi * dtheta) / Iz # Yaw acceleration

    # State derivative dX/dt
    dXdt = np.zeros(12)
    dXdt[0:6] = X[6:12] # [dx, dy, dz, dphi, dtheta, dpsi]
    dXdt[6:12] = [ddx, ddy, ddz, ddphi, ddtheta, ddpsi]
    
    return dXdt

# ==============================================================================
# 2. PERFORMANCE METRICS AND COST FUNCTION (Fitness)
# ==============================================================================

def calculate_metrics(t, z, z_des, errors):
    """Calculates all performance metrics based on the simulation results."""
    
    # --- 1. Root Mean Squared Error (RMSE) ---
    rmse = np.sqrt(np.mean(errors[:, 0]**2)) # Only using altitude error for comparison table

    # --- 2. Overshoot (Mp) --- (CRITICAL: Corrected Logic from Paper Source 404)
    max_z = np.max(z)
    if abs(z_des) < 1e-6:
        # Case: z_des = 0 (or near zero)
        overshoot = max(0, max_z) * 100
    else:
        # Case: z_des != 0
        # Formula: (Max Value - Desired Value) / (|Desired Value|) * 100
        overshoot = max(0, (max_z - z_des) / abs(z_des)) * 100

    # --- 3. Settling Time (ts) ---
    settling_time = TSPAN[1]
    tolerance = 0.02 * abs(z_des) if abs(z_des) > 1e-6 else 0.02 # 2% tolerance
    
    steady_state_start_index = len(z)
    for i in range(len(z) - 1, 0, -1):
        if abs(z[i] - z_des) > tolerance:
            steady_state_start_index = i
            break
            
    # Check if the system ever settled
    if steady_state_start_index < len(z) - 1:
        settling_time = t[steady_state_start_index + 1]
    
    # --- 4. Integral Errors (ITSE, IAE) ---
    # Assuming trapezoidal integration for approximation
    t_abs_err = t * np.abs(errors[:, 0]) # t * |e(t)|
    t_sq_err = t * errors[:, 0]**2 # t * e(t)^2 (ITSE is typically Integral of T*e^2)
    
    iae = np.trapz(np.abs(errors[:, 0]), t)
    itse = np.trapz(t_sq_err, t) # ITSE (Integral Time Squared Error)

    # --- 5. Composite Cost Function I (Fitness) --- (Source 400)
    # I = 0.3*min(ts/10, 1) + 0.3*min(Mp/100, 1) + 0.2*min(ITSE/50, 1) + 0.2*min(IAE/20, 1)
    
    I = 0.3 * min(settling_time / 10.0, 1.0) \
        + 0.3 * min(overshoot / 100.0, 1.0) \
        + 0.2 * min(itse / 50.0, 1.0) \
        + 0.2 * min(iae / 20.0, 1.0)
    
    metrics = {
        'RMSE': rmse,
        'Settling_Time': settling_time,
        'Overshoot': overshoot,
        'ITSE': itse,
        'IAE': iae,
        'Fitness': I
    }
    return metrics

def cost_function(gains, des_refs):
    """
    Fitness function minimized by PSO.
    This function is directly called by the pso solver.
    """
    integral_error_state.reset() # Reset integrator before each run
    
    # Solve the ODE
    try:
        sol = solve_ivp(
            fun=lambda t, X: quadrotor_dynamics(t, X, gains, des_refs),
            t_span=TSPAN,
            y0=X0,
            method='RK45',
            dense_output=True,
            max_step=0.01 # Max step size for simulation stability
        )
        
        # Extract Altitude (z) and Errors (only altitude for cost func)
        t = sol.t
        z = sol.y[2, :]
        z_des = des_refs[0]
        
        # Approximate errors for calculation
        errors_z = z_des - z
        errors = np.array([errors_z]).T 
        
        metrics = calculate_metrics(t, z, z_des, errors)
        
        return metrics['Fitness']

    except Exception:
        # Return a very high cost if the simulation fails (unstable gains)
        return 100.0


# ==============================================================================
# 3. ZIEGLER-NICHOLS BENCHMARK SIMULATION
# ==============================================================================

def simulate_zn_performance(scenario, gains):
    """Simulates one scenario using fixed ZN gains and returns all metrics."""
    des_refs = scenario[['z_des', 'phi_des', 'theta_des', 'psi_des']].values
    
    integral_error_state.reset()
    
    try:
        sol = solve_ivp(
            fun=lambda t, X: quadrotor_dynamics(t, X, gains, des_refs),
            t_span=TSPAN,
            y0=X0,
            method='RK45',
            dense_output=True,
            max_step=0.01
        )

        t = sol.t
        z = sol.y[2, :]
        z_des = des_refs[0]
        errors = np.array([z_des - z]).T 
        
        metrics = calculate_metrics(t, z, z_des, errors)
        return metrics
    
    except Exception:
        # Return NaN for metrics if the ZN simulation is unstable
        return {key: np.nan for key in ['RMSE', 'Settling_Time', 'Overshoot', 'ITSE', 'IAE', 'Fitness']}


# ==============================================================================
# 4. PSO OPTIMIZATION (STATISTICAL RUNS)
# ==============================================================================

def optimize_test_scenario(scenario, num_runs=30):
    """Runs PSO optimization N times (30 for statistical analysis)."""
    print(f"--- Running PSO for {scenario['Test_Maneuver']} ({num_runs} times) ---")
    
    des_refs = scenario[['z_des', 'phi_des', 'theta_des', 'psi_des']].values
    
    # PSO Configuration (Table 2 - 12 parameters to optimize)
    # The search bounds (lb, ub) are critical. Using example bounds here.
    # Note: Kp,z bounds are typically higher than Kp_phi/theta/psi.
    kp_z_bounds = [2.0, 15.0]
    ki_z_bounds = [0.0, 1.0]
    kd_z_bounds = [0.0, 5.0] # K_d=5.0 was maxed in Table 4

    kp_angle_bounds = [0.1, 10.0]
    ki_angle_bounds = [0.0, 0.5]
    kd_angle_bounds = [0.0, 5.0]

    # Lower bound (12 elements: Kp, Ki, Kd for z, phi, theta, psi)
    lb = np.array([kp_z_bounds[0], ki_z_bounds[0], kd_z_bounds[0]] * 4)
    # Upper bound (12 elements)
    ub = np.array([kp_z_bounds[1], ki_z_bounds[1], kd_z_bounds[1]] * 4)
    
    # Adjust for angle control if needed (as stated in Source 422)
    lb[3:6] = [kp_angle_bounds[0], ki_angle_bounds[0], kd_angle_bounds[0]] # Phi
    ub[3:6] = [kp_angle_bounds[1], ki_angle_bounds[1], kd_angle_bounds[1]]
    lb[6:9] = [kp_angle_bounds[0], ki_angle_bounds[0], kd_angle_bounds[0]] # Theta
    ub[6:9] = [kp_angle_bounds[1], ki_angle_bounds[1], kd_angle_bounds[1]]
    lb[9:12] = [kp_angle_bounds[0], ki_angle_bounds[0], kd_angle_bounds[0]] # Psi
    ub[9:12] = [kp_angle_bounds[1], ki_angle_bounds[1], kd_angle_bounds[1]]

    # PSO parameters (Table 2)
    swarmsize = 50
    maxiter = 100
    
    all_pso_results = []

    for run in range(num_runs):
        # Use lambda to pass required arguments to cost_function
        xopt, fopt = pso(cost_function, lb, ub, 
                         args=(des_refs,), 
                         swarmsize=swarmsize, 
                         maxiter=maxiter, 
                         minfunc=1e-5, # Stop if fitness difference is small
                         debug=False)
        
        # After optimization, evaluate the final best gains one last time
        final_metrics = simulate_zn_performance(scenario, xopt)
        
        result = {
            'Run': run + 1,
            'Best_Fitness': fopt,
            'Kp_z': xopt[0], 'Ki_z': xopt[1], 'Kd_z': xopt[2],
            'Kp_phi': xopt[3], 'Ki_phi': xopt[4], 'Kd_phi': xopt[5],
            'Kp_theta': xopt[6], 'Ki_theta': xopt[7], 'Kd_theta': xopt[8],
            'Kp_psi': xopt[9], 'Ki_psi': xopt[10], 'Kd_psi': xopt[11],
            **final_metrics
        }
        all_pso_results.append(result)

    return pd.DataFrame(all_pso_results)


# ==============================================================================
# 5. MAIN COMPARISON AND DATA COLLECTION
# ==============================================================================

def main_comparison():
    # --- A. Test Scenarios (Table 3) ---
    scenarios = pd.DataFrame({
        'Test_Maneuver': [
            'Stationary Takeoff', 'Lateral Drift Takeoff', 
            'Inclined Climb-Out', 'Yaw-Controlled Takeoff', 
            'Transitional Takeoff with Attitude Shift'
        ],
        'z_des': [1.0, 1.5, 2.0, 1.0, 0.5],
        'phi_des': [0.0, 0.1, -0.2, 0.0, -0.1],
        'theta_des': [0.0, -0.1, 0.2, 0.0, -0.1],
        'psi_des': [0.0, 0.0, 0.0, pi/4, pi/4]
    })
    
    # --- B. Ziegler-Nichols Benchmark (Placeholder/Example) ---
    # In a real ZN test, you would derive these gains. 
    # Since the paper doesn't list the ZN gains, we use a placeholder set.
    # The paper implies ZN tuning might use the same gains for all axes.
    # Using a typical ZN result for altitude (z) and applying it to all 4 axes:
    
    # K_u=20, P_u=1.2 -> ZN PID: Kp=12, Ki=20, Kd=1.8 (Using the standard ZN formula)
    Kp_zn = 12.0; Ki_zn = 20.0; Kd_zn = 1.8
    zn_gains = np.array([Kp_zn, Ki_zn, Kd_zn] * 4) 
    
    zn_results = []
    print("\n================== 1. ZN Benchmark Simulation ==================")
    for index, scenario in scenarios.iterrows():
        metrics = simulate_zn_performance(scenario, zn_gains)
        result = {'Maneuver': scenario['Test_Maneuver'], 'Tuning_Method': 'ZN', **metrics}
        zn_results.append(result)
        print(f"ZN Performance for {scenario['Test_Maneuver']}: RMSE={metrics['RMSE']:.4f}")
    
    zn_df = pd.DataFrame(zn_results)
    
    # --- C. PSO Statistical Optimization ---
    pso_results_list = []
    
    print("\n================ 2. PSO Optimization Runs (30x) ================")
    for index, scenario in scenarios.iterrows():
        pso_df = optimize_test_scenario(scenario, num_runs=30)
        pso_df['Maneuver'] = scenario['Test_Maneuver']
        pso_df['Tuning_Method'] = 'PSO'
        pso_results_list.append(pso_df)
        
        # Calculate statistical summary for the current scenario (matches Table 5)
        mean_fitness = pso_df['Best_Fitness'].mean()
        std_dev = pso_df['Best_Fitness'].std()
        print(f"PSO Final Metrics Summary: Mean Fitness={mean_fitness:.4f}, Std Dev={std_dev:.4f}")
        
    pso_full_df = pd.concat(pso_results_list, ignore_index=True)
    
    # --- D. Final Statistical Summary for Comparison (Matches Table 6) ---
    pso_summary = pso_full_df.groupby('Maneuver')[['RMSE', 'Settling_Time', 'Overshoot']].mean().reset_index()
    pso_summary.rename(columns={'RMSE': 'RMSE_PSO', 'Settling_Time': 'ts_PSO', 'Overshoot': 'Mp_PSO'}, inplace=True)
    
    zn_comparison = zn_df[['Maneuver', 'RMSE', 'Settling_Time', 'Overshoot']].copy()
    zn_comparison.rename(columns={'RMSE': 'RMSE_ZN', 'Settling_Time': 'ts_ZN', 'Overshoot': 'Mp_ZN'}, inplace=True)
    
    # Merge and calculate Improvement (%)
    final_comparison = pd.merge(pso_summary, zn_comparison, on='Maneuver')
    final_comparison['RMSE_Improvement (%)'] = ((final_comparison['RMSE_ZN'] - final_comparison['RMSE_PSO']) / final_comparison['RMSE_ZN']) * 100
    
    # --- E. Save Results ---
    pso_full_df.to_csv('PSO_Full_Results_30_Runs.csv', index=False)
    final_comparison.to_csv('ZN_vs_PSO_Statistical_Comparison.csv', index=False)

    print("\n================== 3. Comparison Complete ===================")
    print(final_comparison)
    print("\nFull PSO results saved to 'PSO_Full_Results_30_Runs.csv'")
    print("Statistical summary saved to 'ZN_vs_PSO_Statistical_Comparison.csv'")
    
    return pso_full_df, final_comparison

if __name__ == '__main__':
    # Running the full comparison
    pso_data, comparison_summary = main_comparison()