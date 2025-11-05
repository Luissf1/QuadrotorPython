import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_disturbance_visualization():
    """
    Create a professional visualization of the disturbance equations
    for your methodology slide.
    """
    # Create time vector
    t = np.linspace(0, 20, 1000)
    
    # Calculate disturbance components
    Fx = 0.5 * np.sin(0.5 * t)
    Fy = 0.5 * np.cos(0.5 * t)
    Fz = np.zeros_like(t)  # No vertical disturbance
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Time domain plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, Fx, 'r-', linewidth=2, label=r'$F_x = 0.5 \cdot \sin(0.5t)$')
    ax1.plot(t, Fy, 'b-', linewidth=2, label=r'$F_y = 0.5 \cdot \cos(0.5t)$')
    ax1.plot(t, Fz, 'g-', linewidth=2, label=r'$F_z = 0$ (No vertical disturbance)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Wind Disturbance Components Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # CORREGIDO: Highlight disturbance period (2.4-7.4 seconds)
    ax1.axvspan(2.4, 7.4, alpha=0.2, color='yellow', label='Disturbance Period (2.4-7.4s)')
    
    # Add vertical lines to mark boundaries
    ax1.axvline(x=2.4, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=7.4, color='orange', linestyle='--', alpha=0.7)
    
    # 2. Vector field visualization (at a specific time)
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create grid for vector field
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    
    # Calculate vector field at t=5 seconds (within disturbance period)
    t_sample = 5
    Fx_sample = 0.5 * np.sin(0.5 * t_sample) * np.ones_like(X)
    Fy_sample = 0.5 * np.cos(0.5 * t_sample) * np.ones_like(Y)
    
    # Plot vector field
    ax2.quiver(X, Y, Fx_sample, Fy_sample, scale=5, color='purple', alpha=0.7)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Disturbance Vector Field at t = {t_sample}s (Within 2.4-7.4s period)')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3)
    
    # Add text annotation with equations
    equation_text = (
        r'$\mathbf{F_d} = [F_x, F_y, F_z]^T = $'
        r'$[0.5 \sin(0.5t), 0.5 \cos(0.5t), 0]^T$'
    )
    ax2.text(0.5, -1.5, equation_text, fontsize=11, 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8))
    
    # 3. Polar plot showing the rotating nature of the disturbance
    ax3 = fig.add_subplot(gs[1, 1], polar=True)
    
    # Calculate magnitude and angle
    magnitude = np.sqrt(Fx**2 + Fy**2)
    angle = np.arctan2(Fy, Fx)
    
    # Plot for a few cycles
    cycles_to_show = 2
    t_show = t[t <= cycles_to_show * 4 * np.pi]  # Show 2 full cycles (4π each)
    angle_show = angle[t <= cycles_to_show * 4 * np.pi]
    mag_show = magnitude[t <= cycles_to_show * 4 * np.pi]
    
    ax3.plot(angle_show, mag_show, 'm-', linewidth=2)
    ax3.set_title('Rotating Nature of Wind Disturbance', pad=20)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('disturbance_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_simplified_disturbance_plot():
    """
    Create a simplified version of the disturbance plot for your presentation slide.
    """
    # Create time vector
    t = np.linspace(0, 20, 1000)
    
    # Calculate disturbance components
    Fx = 0.5 * np.sin(0.5 * t)
    Fy = 0.5 * np.cos(0.5 * t)
    
    # Create simplified figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Time domain plot
    ax1.plot(t, Fx, 'r-', linewidth=2, label=r'$F_x = 0.5 \cdot \sin(0.5t)$')
    ax1.plot(t, Fy, 'b-', linewidth=2, label=r'$F_y = 0.5 \cdot \cos(0.5t)$')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Wind Disturbance Components')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # CORREGIDO: Highlight disturbance period (2.4-7.4 seconds)
    ax1.axvspan(2.4, 7.4, alpha=0.2, color='yellow', label='Disturbance Period (2.4-7.4s)')
    ax1.axvline(x=2.4, color='orange', linestyle='--', alpha=0.7)
    ax1.axvline(x=7.4, color='orange', linestyle='--', alpha=0.7)
    
    # Right: Vector diagram at specific time
    t_sample = 5  # Within the disturbance period
    Fx_sample = 0.5 * np.sin(0.5 * t_sample)
    Fy_sample = 0.5 * np.cos(0.5 * t_sample)
    
    ax2.quiver(0, 0, Fx_sample, Fy_sample, angles='xy', scale_units='xy', scale=1, 
               color='purple', width=0.015)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlabel('X Force Component')
    ax2.set_ylabel('Y Force Component')
    ax2.set_title(f'Disturbance Vector at t = {t_sample}s\n(Within 2.4-7.4s period)')
    ax2.grid(True, alpha=0.3)
    
    # Add text with equations
    equation_text = (
        r'$\mathbf{F_d} = [F_x, F_y, F_z]^T = $'
        r'$[0.5 \sin(0.5t), 0.5 \cos(0.5t), 0]^T$'
    )
    ax2.text(0, -0.7, equation_text, fontsize=11, 
             ha='center', va='center', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('disturbance_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_methodology_visualization():
    """
    Create a combined visualization showing both PSO-PID optimization 
    and disturbance modeling for your methodology slide.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: PSO-PID optimization diagram (simplified)
    ax1.text(0.5, 0.9, "PSO Optimization", ha='center', va='center', 
             fontsize=14, fontweight='bold', transform=ax1.transAxes)
    
    # Draw simplified PSO-PID diagram
    ax1.plot([0.2, 0.8], [0.7, 0.7], 'k-', lw=2)  # Top line
    ax1.plot([0.2, 0.8], [0.3, 0.3], 'k-', lw=2)  # Bottom line
    
    # Add blocks
    ax1.text(0.3, 0.7, "PSO\nAlgorithm", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    ax1.text(0.7, 0.7, "PID\nController", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
    ax1.text(0.5, 0.3, "Quadrotor\nDynamics", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral"))
    
    # Add arrows
    ax1.arrow(0.35, 0.65, 0.3, -0.3, head_width=0.02, head_length=0.02, 
              fc='k', ec='k', length_includes_head=True)
    ax1.arrow(0.65, 0.35, -0.3, 0.3, head_width=0.02, head_length=0.02, 
              fc='k', ec='k', length_includes_head=True)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("PSO-PID Optimization Process", fontsize=12)
    
    # Right: Disturbance model
    t = np.linspace(0, 15, 1000)
    Fx = 0.5 * np.sin(0.5 * t)
    Fy = 0.5 * np.cos(0.5 * t)
    
    ax2.plot(t, Fx, 'r-', linewidth=2, label=r'$F_x = 0.5 \cdot \sin(0.5t)$')
    ax2.plot(t, Fy, 'b-', linewidth=2, label=r'$F_y = 0.5 \cdot \cos(0.5t)$')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force (N)')
    ax2.set_title("Wind Disturbance Model (Period: 2.4-7.4s)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # CORREGIDO: Highlight the correct period
    ax2.axvspan(2.4, 7.4, alpha=0.2, color='yellow')
    ax2.axvline(x=2.4, color='orange', linestyle='--', alpha=0.7)
    ax2.axvline(x=7.4, color='orange', linestyle='--', alpha=0.7)
    
    # Add equation text
    equation_text = (
        r'$\mathbf{F_d} = [F_x, F_y, F_z]^T = $'
        r'$[0.5 \sin(0.5t), 0.5 \cos(0.5t), 0]^T$'
    )
    ax2.text(0.5, -0.3, equation_text, fontsize=11, ha='center', va='center', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('methodology_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_controller_comparison():
    """
    Create a comparison of controller performance with and without disturbance.
    CORRECTED VERSION: Uses 2.4-7.4s disturbance period
    """
    # Simulate some example data for comparison
    t = np.linspace(0, 10, 1000)
    
    # Find indices for the disturbance period (2.4-7.4s)
    disturbance_start_idx = np.argmin(np.abs(t - 2.4))
    disturbance_end_idx = np.argmin(np.abs(t - 7.4))
    
    # Simulated responses
    z_desired = np.ones_like(t)  # Desired height of 1m
    
    # Standard controller response (underdamped with disturbance)
    z_standard = 1 - np.exp(-0.8*t) * (np.cos(2*t) + 0.4*np.sin(2*t))
    # CORREGIDO: Add disturbance effect during 2.4-7.4s
    z_standard[disturbance_start_idx:disturbance_end_idx] += 0.3 * np.sin(0.5*(t[disturbance_start_idx:disturbance_end_idx]-2.4))
    
    # Disturbance-optimized controller response (better disturbance rejection)
    z_optimized = 1 - np.exp(-1.2*t) * (np.cos(2.5*t) + 0.3*np.sin(2.5*t))
    # CORREGIDO: Add smaller disturbance effect during 2.4-7.4s
    z_optimized[disturbance_start_idx:disturbance_end_idx] += 0.1 * np.sin(0.5*(t[disturbance_start_idx:disturbance_end_idx]-2.4))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot responses
    plt.plot(t, z_desired, 'k--', linewidth=2, label='Desired Height')
    plt.plot(t, z_standard, 'r-', linewidth=2, label='Standard PSO-PID')
    plt.plot(t, z_optimized, 'b-', linewidth=2, label='Disturbance-Optimized PSO-PID')
    
    # CORREGIDO: Highlight the correct disturbance period
    plt.axvspan(2.4, 7.4, alpha=0.2, color='yellow', label='Disturbance Period (2.4-7.4s)')
    plt.axvline(x=2.4, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=7.4, color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Height z (m)', fontsize=12, fontweight='bold')
    plt.title('Controller Performance Comparison with Wind Disturbance (2.4-7.4s)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add performance metrics text
    plt.text(8, 0.3, 'Performance during disturbance:\n• Standard PSO-PID: Larger oscillations\n• Optimized PSO-PID: Better rejection\n• IAE improvement: 7.41%\n• ITSE improvement: 7.01%', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
             fontsize=10)
    
    plt.savefig('controller_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating disturbance visualization...")
    create_disturbance_visualization()
    
    print("Creating combined methodology visualization...")
    create_combined_methodology_visualization()
    
    print("Creating controller comparison visualization...")
    create_controller_comparison()
    
    print("All visualizations created successfully!")