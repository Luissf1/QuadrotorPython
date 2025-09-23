import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.patches as patches
from matplotlib.patches import Circle
import os

class AdvancedDroneAnimator:
    def __init__(self, trajectory, attitude, time, desired_position=None):
        """
        Advanced drone animator with proper 3D rotation
        
        Parameters:
        - trajectory: numpy array of shape (3, n_points) - [x, y, z] positions
        - attitude: numpy array of shape (3, n_points) - [phi, theta, psi] angles
        - time: numpy array of time points
        - desired_position: desired target position [x_des, y_des, z_des]
        """
        self.trajectory = trajectory
        self.attitude = attitude
        self.time = time
        self.desired_position = desired_position
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 6))
        
        # 3D view
        self.ax_3d = self.fig.add_subplot(131, projection='3d')
        
        # Side view (X-Z plane)
        self.ax_side = self.fig.add_subplot(132)
        
        # Top view (X-Y plane)
        self.ax_top = self.fig.add_subplot(133)
        
        self.setup_plots()
        
        # Animation elements
        self.drone_artists_3d = []
        self.drone_artists_side = []
        self.drone_artists_top = []
        self.trajectory_lines = []
        self.info_text = None
        
    def setup_plots(self):
        """Setup the plot axes and labels"""
        # 3D plot setup
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Drone Flight Animation')
        self.ax_3d.grid(True)
        
        # Side view setup (X-Z plane)
        self.ax_side.set_xlabel('X (m)')
        self.ax_side.set_ylabel('Z (m)')
        self.ax_side.set_title('Side View (X-Z)')
        self.ax_side.grid(True)
        self.ax_side.set_aspect('equal')
        
        # Top view setup (X-Y plane)
        self.ax_top.set_xlabel('X (m)')
        self.ax_top.set_ylabel('Y (m)')
        self.ax_top.set_title('Top View (X-Y)')
        self.ax_top.grid(True)
        self.ax_top.set_aspect('equal')
        
    def euler_to_rotation_matrix(self, phi, theta, psi):
        """
        Convert Euler angles to rotation matrix (Z-Y-X convention)
        """
        # Rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        
        # Combined rotation: R = R_z * R_y * R_x
        return R_z @ R_y @ R_x
    
    def create_drone_model(self):
        """
        Create a detailed 3D drone model
        """
        # Drone body (central rectangular prism)
        body_length, body_width, body_height = 0.15, 0.15, 0.05
        body_vertices = np.array([
            # Bottom face
            [-body_length/2, -body_width/2, -body_height/2],
            [ body_length/2, -body_width/2, -body_height/2],
            [ body_length/2,  body_width/2, -body_height/2],
            [-body_length/2,  body_width/2, -body_height/2],
            # Top face
            [-body_length/2, -body_width/2,  body_height/2],
            [ body_length/2, -body_width/2,  body_height/2],
            [ body_length/2,  body_width/2,  body_height/2],
            [-body_length/2,  body_width/2,  body_height/2]
        ])
        
        # Body faces
        body_faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        
        # Arm parameters
        arm_length = 0.4
        arm_radius = 0.02
        arm_directions = np.array([
            [1, 0, 0],   # front arm
            [-1, 0, 0],  # back arm
            [0, 1, 0],   # right arm
            [0, -1, 0]   # left arm
        ])
        
        # Propeller parameters
        prop_radius = 0.1
        prop_positions = arm_directions * (arm_length / 2)
        
        return {
            'body_vertices': body_vertices,
            'body_faces': body_faces,
            'arm_directions': arm_directions,
            'arm_length': arm_length,
            'prop_positions': prop_positions,
            'prop_radius': prop_radius
        }
    
    def draw_drone_3d(self, ax, position, attitude, color='lightblue'):
        """Draw the drone in 3D with proper rotation"""
        x, y, z = position
        phi, theta, psi = attitude
        
        # Clear previous drone artists
        for artist in self.drone_artists_3d:
            if hasattr(artist, 'remove'):
                artist.remove()
        self.drone_artists_3d = []
        
        # Get rotation matrix
        R_matrix = self.euler_to_rotation_matrix(phi, theta, psi)
        
        # Get drone model
        model = self.create_drone_model()
        body_vertices = model['body_vertices']
        body_faces = model['body_faces']
        arm_directions = model['arm_directions']
        arm_length = model['arm_length']
        prop_positions = model['prop_positions']
        prop_radius = model['prop_radius']
        
        # Rotate and translate body vertices
        body_rotated = body_vertices @ R_matrix.T
        body_translated = body_rotated + position
        
        # Draw body as a simple cube (using scatter for simplicity)
        body_center = position
        body_scatter = ax.scatter([body_center[0]], [body_center[1]], [body_center[2]], 
                                c=color, s=500, marker='o', alpha=0.7)
        self.drone_artists_3d.append(body_scatter)
        
        # Draw arms
        for direction in arm_directions:
            arm_start = position
            arm_end_local = direction * (arm_length / 2)
            arm_end_rotated = arm_end_local @ R_matrix.T
            arm_end = position + arm_end_rotated
            
            arm_line, = ax.plot([arm_start[0], arm_end[0]], 
                              [arm_start[1], arm_end[1]], 
                              [arm_start[2], arm_end[2]], 
                              'k-', linewidth=4)
            self.drone_artists_3d.append(arm_line)
            
            # Draw propeller at arm end
            prop_scatter = ax.scatter([arm_end[0]], [arm_end[1]], [arm_end[2]], 
                                    c='green', s=200, marker='o', alpha=0.8)
            self.drone_artists_3d.append(prop_scatter)
        
        return self.drone_artists_3d
    
    def draw_drone_2d(self, ax, position, attitude, view='side'):
        """Draw drone in 2D view (side or top)"""
        x, y, z = position
        phi, theta, psi = attitude
        
        # Clear previous artists
        artists = self.drone_artists_side if view == 'side' else self.drone_artists_top
        for artist in artists:
            if hasattr(artist, 'remove'):
                artist.remove()
        
        new_artists = []
        
        if view == 'side':  # X-Z view
            # Draw drone body
            body = ax.scatter(x, z, c='lightblue', s=200, alpha=0.8)
            new_artists.append(body)
            
            # Draw orientation indicator
            arrow_length = 0.3
            dx = arrow_length * np.cos(psi)
            dz = arrow_length * np.sin(theta)
            
            arrow = ax.arrow(x, z, dx, dz, head_width=0.05, head_length=0.1, 
                           fc='red', ec='red')
            new_artists.append(arrow)
            
        else:  # Top view (X-Y)
            # Draw drone body
            body = ax.scatter(x, y, c='lightblue', s=200, alpha=0.8)
            new_artists.append(body)
            
            # Draw orientation indicator (yaw)
            arrow_length = 0.3
            dx = arrow_length * np.cos(psi)
            dy = arrow_length * np.sin(psi)
            
            arrow = ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, 
                           fc='red', ec='red')
            new_artists.append(arrow)
        
        if view == 'side':
            self.drone_artists_side = new_artists
        else:
            self.drone_artists_top = new_artists
            
        return new_artists
    
    def clear_axes(self, ax):
        """Safe method to clear axes compatible with different matplotlib versions"""
        # Remove artists one by one
        for artist in ax.lines + ax.collections + ax.patches + ax.texts:
            try:
                artist.remove()
            except:
                pass
        
        # Clear text
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if hasattr(ax, 'set_zlabel'):
            ax.set_zlabel('')
    
    def plot_trajectory(self):
        """Plot the complete trajectory"""
        # Clear previous trajectory lines
        for line in self.trajectory_lines:
            if hasattr(line, 'remove'):
                line.remove()
        self.trajectory_lines = []
        
        # 3D trajectory
        line_3d = self.ax_3d.plot(self.trajectory[0], self.trajectory[1], self.trajectory[2], 
                                'b-', alpha=0.5, linewidth=1, label='Flight Path')[0]
        self.trajectory_lines.append(line_3d)
        
        # Side view trajectory
        line_side = self.ax_side.plot(self.trajectory[0], self.trajectory[2], 
                                    'b-', alpha=0.5, linewidth=1, label='Path')[0]
        self.trajectory_lines.append(line_side)
        
        # Top view trajectory  
        line_top = self.ax_top.plot(self.trajectory[0], self.trajectory[1], 
                                  'b-', alpha=0.5, linewidth=1, label='Path')[0]
        self.trajectory_lines.append(line_top)
        
        # Plot desired position if provided
        if self.desired_position is not None:
            x_des, y_des, z_des = self.desired_position
            
            # 3D target
            target_3d = self.ax_3d.scatter([x_des], [y_des], [z_des], c='red', s=200, 
                                         marker='*', label='Target')
            self.trajectory_lines.append(target_3d)
            
            # Side view target
            target_side = self.ax_side.scatter([x_des], [z_des], c='red', s=200, 
                                             marker='*', label='Target')
            self.trajectory_lines.append(target_side)
            
            # Top view target
            target_top = self.ax_top.scatter([x_des], [y_des], c='red', s=200, 
                                           marker='*', label='Target')
            self.trajectory_lines.append(target_top)
    
    def update_animation(self, frame):
        """Update animation for each frame"""
        try:
            # Get current state
            position = self.trajectory[:, frame]
            attitude = self.attitude[:, frame]
            current_time = self.time[frame]
            
            # Clear only drone elements, keep trajectories
            for artist in self.drone_artists_3d + self.drone_artists_side + self.drone_artists_top:
                if hasattr(artist, 'remove'):
                    artist.remove()
            
            self.drone_artists_3d = []
            self.drone_artists_side = []
            self.drone_artists_top = []
            
            # Draw drone in all views
            self.draw_drone_3d(self.ax_3d, position, attitude)
            self.draw_drone_2d(self.ax_side, position, attitude, 'side')
            self.draw_drone_2d(self.ax_top, position, attitude, 'top')
            
            # Update titles
            x, y, z = position
            phi, theta, psi = np.degrees(attitude)
            
            self.ax_3d.set_title(f'3D Flight - Time: {current_time:.1f}s\nPos: ({x:.2f}, {y:.2f}, {z:.2f})m')
            self.ax_side.set_title(f'Side View - Time: {current_time:.1f}s')
            self.ax_top.set_title(f'Top View - Time: {current_time:.1f}s')
            
            # Set consistent axis limits with margin
            margin = 2.0
            x_min, x_max = np.min(self.trajectory[0])-margin, np.max(self.trajectory[0])+margin
            y_min, y_max = np.min(self.trajectory[1])-margin, np.max(self.trajectory[1])+margin
            z_min, z_max = 0, np.max(self.trajectory[2])+margin  # Start from ground
            
            self.ax_3d.set_xlim(x_min, x_max)
            self.ax_3d.set_ylim(y_min, y_max)
            self.ax_3d.set_zlim(z_min, z_max)
            
            self.ax_side.set_xlim(x_min, x_max)
            self.ax_side.set_ylim(z_min, z_max)
            
            self.ax_top.set_xlim(x_min, x_max)
            self.ax_top.set_ylim(y_min, y_max)
            
            all_artists = (self.drone_artists_3d + self.drone_artists_side + 
                          self.drone_artists_top + self.trajectory_lines)
            
            return [artist for artist in all_artists if hasattr(artist, 'set_visible')]
            
        except Exception as e:
            print(f"Error in frame {frame}: {e}")
            return []
    
    def init_animation(self):
        """Initialize animation"""
        self.plot_trajectory()
        return self.trajectory_lines
    
    def animate(self, interval=50, save=False, filename='drone_animation.gif'):
        """Create and run the animation"""
        # Plot initial trajectory
        self.plot_trajectory()
        
        # Set initial axis limits
        margin = 2.0
        x_min, x_max = np.min(self.trajectory[0])-margin, np.max(self.trajectory[0])+margin
        y_min, y_max = np.min(self.trajectory[1])-margin, np.max(self.trajectory[1])+margin
        z_min, z_max = 0, np.max(self.trajectory[2])+margin
        
        self.ax_3d.set_xlim(x_min, x_max)
        self.ax_3d.set_ylim(y_min, y_max)
        self.ax_3d.set_zlim(z_min, z_max)
        self.ax_side.set_xlim(x_min, x_max)
        self.ax_side.set_ylim(z_min, z_max)
        self.ax_top.set_xlim(x_min, x_max)
        self.ax_top.set_ylim(y_min, y_max)
        
        # Create animation with blit=True for better performance
        anim = FuncAnimation(
            self.fig, self.update_animation, frames=len(self.time),
            init_func=self.init_animation, interval=interval, 
            blit=True, repeat=True, cache_frame_data=False
        )
        
        plt.tight_layout()
        
        # Save animation if requested
        if save:
            try:
                os.makedirs('animations', exist_ok=True)
                filepath = os.path.join('animations', filename)
                print(f"Saving animation to {filepath}...")
                
                # Reduce frames for faster saving
                save_frames = min(100, len(self.time))  # Max 100 frames for GIF
                step = max(1, len(self.time) // save_frames)
                
                # Create a partial animation for saving
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=1000//interval)
                
                anim.save(filepath, writer=writer, dpi=100)
                print("Animation saved successfully!")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print("Showing animation without saving...")
        
        plt.show()
        return anim

# Fixed quadrotor dynamics simulation
def simulate_quadrotor_dynamics(gains, z_des, phi_des, theta_des, psi_des, duration=10):
    """Simulate quadrotor dynamics with PID control - FIXED VERSION"""
    
    def quadrotor_dynamics(t, X, gains, setpoints):
        m, g, Ix, Iy, Iz = 1.0, 9.81, 0.1, 0.1, 0.2
        z_des, phi_des, theta_des, psi_des = setpoints
        
        # Extract gains
        Kp_z, Ki_z, Kd_z = gains[0], gains[1], gains[2]
        Kp_phi, Ki_phi, Kd_phi = gains[3], gains[4], gains[5]
        Kp_theta, Ki_theta, Kd_theta = gains[6], gains[7], gains[8]
        Kp_psi, Ki_psi, Kd_psi = gains[9], gains[10], gains[11]
        
        pos = X[:6]   # [x, y, z, phi, theta, psi]
        vel = X[6:]   # [vx, vy, vz, vphi, vtheta, vpsi]
        
        # Calculate errors
        err_z = z_des - pos[2]
        err_phi = phi_des - pos[3]
        err_theta = theta_des - pos[4]
        err_psi = psi_des - pos[5]
        
        # PID control (simplified)
        U1 = Kp_z * err_z + Kd_z * (-vel[2]) +  m*g  # Add gravity compensation
        U2 = Kp_phi * err_phi + Kd_phi * (-vel[3])
        U3 = Kp_theta * err_theta + Kd_theta * (-vel[4])
        U4 = Kp_psi * err_psi + Kd_psi * (-vel[5])
        
        # Limit control inputs
        U1 = np.clip(U1, 0, 20)
        U2 = np.clip(U2, -5, 5)
        U3 = np.clip(U3, -5, 5)
        U4 = np.clip(U4, -5, 5)
        
        # Dynamics equations (simplified for stability)
        acc_x = (np.sin(pos[4]) * U1) / m  # Simplified for stability
        acc_y = (-np.sin(pos[3]) * U1) / m
        acc_z = (np.cos(pos[3]) * np.cos(pos[4]) * U1) / m - g
        
        acc_phi = U2 / Ix
        acc_theta = U3 / Iy
        acc_psi = U4 / Iz
        
        return np.concatenate((vel, [acc_x, acc_y, acc_z, acc_phi, acc_theta, acc_psi]))
    
    # Initial conditions - start from ground
    X0 = np.zeros(12)
    X0[2] = 0.1  # Start slightly above ground
    
    # Time span with more points for smoother animation
    t_span = (0, duration)
    t_eval = np.linspace(0, duration, min(200, int(duration * 20)))
    
    # Solve ODE with error handling
    setpoints = (z_des, phi_des, theta_des, psi_des)
    
    try:
        sol = solve_ivp(
            lambda t, X: quadrotor_dynamics(t, X, gains, setpoints),
            t_span, X0, t_eval=t_eval, method='RK45', rtol=1e-6
        )
        
        if sol.success:
            return sol.t, sol.y
        else:
            print("ODE solver failed, using fallback trajectory")
            raise Exception("Solver failed")
            
    except:
        # Fallback: create a simple trajectory
        print("Using fallback trajectory")
        t = t_eval
        # Simple smooth trajectory to desired position
        x = 0.5 * np.sin(0.5 * t)
        y = 0.5 * np.cos(0.5 * t)
        z = z_des * (1 - np.exp(-2 * t))  # Smooth approach to desired height
        
        # Smooth attitude changes
        phi = phi_des * (1 - np.exp(-3 * t))
        theta = theta_des * (1 - np.exp(-3 * t))
        psi = psi_des * (1 - np.exp(-2 * t))
        
        X = np.vstack([x, y, z, phi, theta, psi, np.zeros((6, len(t)))])
        return t, X

def create_stable_animation():
    """Create a stable animation with better trajectory"""
    print("Creating stable animation...")
    
    # Stable gains
    stable_gains = np.array([12.0, 1.0, 3.0, 4.0, 0.2, 1.5, 4.0, 0.2, 1.5, 3.0, 0.15, 1.0])
    
    # Simulate a smooth trajectory
    t, X = simulate_quadrotor_dynamics(stable_gains, 2.0, 0.1, -0.1, np.pi/3, duration=12)
    
    # Extract position and attitude
    trajectory = X[:3]  # x, y, z
    attitude = X[3:6]   # phi, theta, psi
    
    print(f"Trajectory simulation completed: {trajectory.shape}")
    print(f"Final position: ({trajectory[0, -1]:.2f}, {trajectory[1, -1]:.2f}, {trajectory[2, -1]:.2f})")
    
    # Create animator
    animator = AdvancedDroneAnimator(trajectory, attitude, t)
    
    # Run animation without saving first to test
    print("Starting animation...")
    anim = animator.animate(interval=60, save=False)  # Test without saving first
    
    return anim, trajectory, attitude, t

def quick_test_animation():
    """Quick test with simple hover"""
    print("Running quick test...")
    gains = np.array([10.0, 0.8, 2.0, 3.0, 0.15, 1.0, 3.0, 0.15, 1.0, 2.0, 0.1, 0.5])
    
    t, X = simulate_quadrotor_dynamics(gains, 1.5, 0.0, 0.0, 0.0, duration=8)
    
    trajectory = X[:3]
    attitude = X[3:6]
    
    animator = AdvancedDroneAnimator(trajectory, attitude, t)
    anim = animator.animate(interval=50, save=False)
    
    return anim

if __name__ == "__main__":
    print("Advanced Drone Animation with Proper Rotation - FIXED VERSION")
    print("=" * 60)
    
    try:
        # Always use the stable version first
        anim, trajectory, attitude, time = create_stable_animation()
        print("Animation completed successfully!")
        
    except Exception as e:
        print(f"Error in main animation: {e}")
        print("Falling back to quick test...")
        try:
            anim = quick_test_animation()
            print("Quick test completed!")
        except Exception as e2:
            print(f"Quick test also failed: {e2}")
            print("Please check your matplotlib installation and try again.")