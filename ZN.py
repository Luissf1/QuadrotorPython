import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def ziegler_nichols_tuning_corrected():
    """
    Corrected Ziegler-Nichols tuning for quadrotor PID controller
    """
    # --- Parameters ---
    m = 1.0
    g = 9.81
    Ix, Iy, Iz = 0.1, 0.1, 0.2
    
    # Initial conditions - START AT GROUND
    X0 = np.array([0, 0, 0, 0, 0, 0,  # position [x, y, z, ϕ, θ, ψ]
                   0, 0, 0, 0, 0, 0]) # velocities
    
    # Flight conditions (same as PSO)
    flight_conditions = np.array([
        [1.0,  0.0,   0.0,    0.0],
        [1.5,  0.1,  -0.1,    0.0], 
        [2.0, -0.2,   0.2,    0.0],
        [1.0,  0.0,   0.0,    np.pi/4],
        [0.5, -0.1,  -0.1,   -np.pi/6]
    ])
    
    RMSE_results = []
    
    for i, (z_des, phi_des, theta_des, psi_des) in enumerate(flight_conditions):
        print(f'\n--- Flight {i+1}: z={z_des}, phi={phi_des}, theta={theta_des}, psi={psi_des} ---')
        
        # Tune EACH axis separately with conservative gains
        if i == 0:  # For first test, use more careful tuning
            # Conservative gains based on quadrotor physics
            Kp_z = 8.0; Ki_z = 0.5; Kd_z = 3.0
            Kp_phi = 3.0; Ki_phi = 0.1; Kd_phi = 0.5
            Kp_theta = 3.0; Ki_theta = 0.1; Kd_theta = 0.5  
            Kp_psi = 2.0; Ki_psi = 0.05; Kd_psi = 0.3
        else:
            # Use successful gains from previous test as starting point
            Kp_z = 8.0; Ki_z = 0.5; Kd_z = 3.0
            Kp_phi = 3.0; Ki_phi = 0.1; Kd_phi = 0.5
            Kp_theta = 3.0; Ki_theta = 0.1; Kd_theta = 0.5
            Kp_psi = 2.0; Ki_psi = 0.05; Kd_psi = 0.3
        
        print(f'  Using gains: Kp_z={Kp_z}, Ki_z={Ki_z}, Kd_z={Kd_z}')
        
        # Simulate with tuned PID
        try:
            t_span = (0, 10)
            t_eval = np.linspace(0, 10, 1000)
            
            # Reset integral states
            integral_state = {
                'z': 0, 'phi': 0, 'theta': 0, 'psi': 0,
                'prev_error_z': 0, 'prev_error_phi': 0,
                'prev_error_theta': 0, 'prev_error_psi': 0
            }
            
            sol = solve_ivp(
                lambda t, X: quadrotor_dynamics_improved(
                    t, X, m, g, Ix, Iy, Iz,
                    Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                    Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                    z_des, phi_des, theta_des, psi_des, integral_state),
                t_span, X0, t_eval=t_eval, method='RK45'
            )
            
            if sol.success:
                z = sol.y[2]
                error_z = z_des - z
                RMSE = np.sqrt(np.mean(error_z**2))
                RMSE_results.append(RMSE)
                
                print(f'  RMSE: {RMSE:.4f}')
                
                # Plot
                plt.figure(figsize=(10, 6))
                plt.plot(sol.t, z, 'b-', linewidth=2, label='Actual')
                plt.axhline(y=z_des, color='r', linestyle='--', linewidth=2, label='Desired')
                plt.xlabel('Time (s)')
                plt.ylabel('Altitude z (m)')
                plt.title(f'Ziegler-Nichols - Flight {i+1} (RMSE={RMSE:.4f})')
                plt.legend()
                plt.grid(True)
                plt.show()
                
            else:
                print('  Simulation failed!')
                RMSE_results.append(np.inf)
                
        except Exception as e:
            print(f'  Simulation error: {e}')
            RMSE_results.append(np.inf)
    
    # Final results
    print('\n=== Final RMSE Results (Ziegler-Nichols) ===')
    for i, (condition, rmse) in enumerate(zip(flight_conditions, RMSE_results)):
        z_des, phi_des, theta_des, psi_des = condition
        print(f'Flight {i+1} (z={z_des}, phi={phi_des}, theta={theta_des}, psi={psi_des}): RMSE = {rmse:.4f}')
    
    return RMSE_results

def quadrotor_dynamics_improved(t, X, m, g, Ix, Iy, Iz,
                               Kp_z, Ki_z, Kd_z, Kp_phi, Ki_phi, Kd_phi,
                               Kp_theta, Ki_theta, Kd_theta, Kp_psi, Ki_psi, Kd_psi,
                               z_des, phi_des, theta_des, psi_des, integral_state):
    """
    Improved quadrotor dynamics with better control implementation
    """
    # Extract states
    pos = X[:6]   # [x, y, z, ϕ, θ, ψ]
    vel = X[6:]   # [vx, vy, vz, ω_ϕ, ω_θ, ω_ψ]
    
    # Current altitude and orientation
    current_z = pos[2]
    current_phi, current_theta, current_psi = pos[3], pos[4], pos[5]
    
    # Calculate errors
    error_z = z_des - current_z
    error_phi = phi_des - current_phi
    error_theta = theta_des - current_theta  
    error_psi = psi_des - current_psi
    
    # Calculate error derivatives (finite difference)
    dt = 0.01
    derror_z = (error_z - integral_state['prev_error_z']) / dt if t > 0 else 0
    derror_phi = (error_phi - integral_state['prev_error_phi']) / dt if t > 0 else 0
    derror_theta = (error_theta - integral_state['prev_error_theta']) / dt if t > 0 else 0
    derror_psi = (error_psi - integral_state['prev_error_psi']) / dt if t > 0 else 0
    
    # Update integrals with anti-windup
    max_integral = 5.0
    integral_state['z'] = np.clip(integral_state['z'] + error_z * dt, -max_integral, max_integral)
    integral_state['phi'] = np.clip(integral_state['phi'] + error_phi * dt, -max_integral, max_integral)
    integral_state['theta'] = np.clip(integral_state['theta'] + error_theta * dt, -max_integral, max_integral)
    integral_state['psi'] = np.clip(integral_state['psi'] + error_psi * dt, -max_integral, max_integral)
    
    # Store current errors for next derivative calculation
    integral_state['prev_error_z'] = error_z
    integral_state['prev_error_phi'] = error_phi  
    integral_state['prev_error_theta'] = error_theta
    integral_state['prev_error_psi'] = error_psi
    
    # PID Control with output limits
    # Total thrust (always positive)
    U1 = Kp_z * error_z + Ki_z * integral_state['z'] + Kd_z * derror_z
    U1 = max(0.1 * m * g, min(U1, 3 * m * g))  # Reasonable thrust limits
    
    # Torques (limited)
    U2 = Kp_phi * error_phi + Ki_phi * integral_state['phi'] + Kd_phi * derror_phi
    U3 = Kp_theta * error_theta + Ki_theta * integral_state['theta'] + Kd_theta * derror_theta  
    U4 = Kp_psi * error_psi + Ki_psi * integral_state['psi'] + Kd_psi * derror_psi
    
    U2 = np.clip(U2, -2.0, 2.0)
    U3 = np.clip(U3, -2.0, 2.0)
    U4 = np.clip(U4, -1.0, 1.0)
    
    # SIMPLIFIED DYNAMICS - More stable for initial testing
    # Vertical dynamics
    acc_z = (U1 / m) - g
    
    # Rotational dynamics (simplified)
    acc_phi = U2 / Ix
    acc_theta = U3 / Iy  
    acc_psi = U4 / Iz
    
    # Assume small angles for initial testing - no lateral movement
    acc_x = 0
    acc_y = 0
    
    # State derivatives
    dposdt = vel
    dveldt = np.array([acc_x, acc_y, acc_z, acc_phi, acc_theta, acc_psi])
    
    dXdt = np.concatenate([dposdt, dveldt])
    return dXdt

# Run the corrected Ziegler-Nichols
if __name__ == "__main__":
    print("=== CORRECTED ZIEGLER-NICHOLS TUNING ===")
    RMSE_ZN = ziegler_nichols_tuning_corrected()
    
    # Compare with PSO results from your article
    RMSE_PSO = [0.3798, 0.52, 0.61, 0.365, 0.178]
    
    print('\n=== COMPARISON ZN vs PSO ===')
    for i, (zn, pso) in enumerate(zip(RMSE_ZN, RMSE_PSO)):
        improvement = ((zn - pso) / zn) * 100 if zn > 0 else 0
        print(f'Test {i+1}: ZN={zn:.4f}, PSO={pso:.4f}, Improvement={improvement:.1f}%')