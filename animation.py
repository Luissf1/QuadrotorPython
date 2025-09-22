import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

class AdvancedDroneAnimator:
    def __init__(self, trajectory, attitude, time):
        self.trajectory = trajectory
        self.attitude = attitude
        self.time = time
        
        self.fig = plt.figure(figsize=(12, 5))
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_2d = self.fig.add_subplot(122)
        
        self.setup_plots()
        
    def setup_plots(self):
        """Setup the plot axes"""
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Drone Animation')
        
        self.ax_2d.set_xlabel('X (m)')
        self.ax_2d.set_ylabel('Z (m)')
        self.ax_2d.set_title('Side View')
        self.ax_2d.grid(True)
        self.ax_2d.set_aspect('equal')
        
    def rotation_matrix(self, phi, theta, psi):
        """Create rotation matrix from Euler angles"""
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
        
        return R_z @ R_y @ R_x
    
    def create_drone_model(self):
        """Create a 3D drone model"""
        # Drone body (central cube)
        body_vertices = np.array([
            [-0.1, -0.1, -0.1],  # 0
            [ 0.1, -0.1, -0.1],  # 1
            [ 0.1,  0.1, -0.1],  # 2
            [-0.1,  0.1, -0.1],  # 3
            [-0.1, -0.1,  0.1],  # 4
            [ 0.1, -0.1,  0.1],  # 5
            [ 0.1,  0.1,  0.1],  # 6
            [-0.1,  0.1,  0.1]   # 7
        ])
        
        # Arm endpoints
        arm_length = 0.4
        arms = np.array([
            [ arm_length, 0, 0],  # front
            [-arm_length, 0, 0],  # back
            [0,  arm_length, 0],  # right
            [0, -arm_length, 0]   # left
        ])
        
        # Propellers (disks at arm ends)
        prop_radius = 0.15
        return body_vertices, arms, prop_radius
    
    def draw_drone(self, ax, position, attitude, color='red'):
        """Draw the drone at given position and attitude"""
        x, y, z = position
        phi, theta, psi = attitude
        
        # Get rotation matrix
        R = self.rotation_matrix(phi, theta, psi)
        
        # Get drone model
        body_vertices, arms, prop_radius = self.create_drone_model()
        
        # Rotate and translate body
        body_rotated = body_vertices @ R.T
        body_translated = body_rotated + position
        
        # Rotate and translate arms
        arms_rotated = arms @ R.T
        arms_translated = arms_rotated + position
        
        # Plot body (as wireframe)
        body_edges = [
            [0,1], [1,2], [2,3], [3,0],  # bottom
            [4,5], [5,6], [6,7], [7,4],  # top
            [0,4], [1,5], [2,6], [3,7]   # sides
        ]
        
        for edge in body_edges:
            points = body_translated[edge]
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', linewidth=2)
        
        # Plot arms
        for arm in arms_translated:
            ax.plot3D([x, arm[0]], [y, arm[1]], [z, arm[2]], 'b-', linewidth=3)
        
        # Plot propellers (simplified as circles)
        for arm in arms_translated:
            # Create circle in propeller plane
            theta_circle = np.linspace(0, 2*np.pi, 20)
            circle_x = prop_radius * np.cos(theta_circle)
            circle_y = prop_radius * np.sin(theta_circle)
            circle_z = np.zeros_like(circle_x)
            circle = np.vstack([circle_x, circle_y, circle_z]).T
            
            # Rotate circle to propeller orientation and translate
            circle_rotated = circle @ R.T
            circle_translated = circle_rotated + arm
            
            ax.plot3D(circle_translated[:, 0], circle_translated[:, 1], 
                     circle_translated[:, 2], 'g-', linewidth=2)
        
        return ax
    
    def animate_flight(self):
        """Create the animation"""
        # Plot trajectory
        self.ax_3d.plot(self.trajectory[0], self.trajectory[1], self.trajectory[2], 
                       'b-', alpha=0.3, label='Trajectory')
        self.ax_2d.plot(self.trajectory[0], self.trajectory[2], 'b-', alpha=0.3)
        
        def update(frame):
            # Clear previous drone
            self.ax_3d.cla()
            self.ax_2d.cla()
            self.setup_plots()
            
            # Replot trajectory
            self.ax_3d.plot(self.trajectory[0], self.trajectory[1], self.trajectory[2], 
                           'b-', alpha=0.3)
            self.ax_2d.plot(self.trajectory[0], self.trajectory[2], 'b-', alpha=0.3)
            
            # Get current state
            position = self.trajectory[:, frame]
            attitude = self.attitude[:, frame]
            
            # Draw drone
            self.draw_drone(self.ax_3d, position, attitude)
            
            # Draw 2D projection
            self.ax_2d.scatter(position[0], position[2], c='red', s=100)
            self.ax_2d.set_xlim(self.trajectory[0].min()-1, self.trajectory[0].max()+1)
            self.ax_2d.set_ylim(self.trajectory[2].min()-1, self.trajectory[2].max()+1)
            
            # Update title
            self.ax_3d.set_title(f'3D Drone Animation (t = {self.time[frame]:.1f}s)')
            
            return []
        
        anim = FuncAnimation(self.fig, update, frames=len(self.time), 
                           interval=50, blit=False)
        plt.tight_layout()
        plt.show()
        return anim
