import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(npz_file_path="11_30_training.npz"):
    """
    Load training data from .npz file
    
    Returns:
        x_train: (n_samples, 12, 3) - 12 actuator tokens with [initial, deformed, difference]
        y_train: (n_samples, 192, 3) - 192 position tokens with [fx, fy, fz]
    """
    from pathlib import Path
    
    # Try multiple possible file locations
    script_dir = Path(__file__).parent
    possible_files = [
        npz_file_path,
        script_dir / npz_file_path,
        script_dir.parent / npz_file_path,
        Path.cwd() / npz_file_path,
    ]
    
    actual_file_path = None
    for filepath in possible_files:
        if Path(filepath).exists():
            actual_file_path = str(filepath)
            break
    
    if actual_file_path is None:
        raise FileNotFoundError(f"Could not find {npz_file_path} in any of the searched locations")
    
    data = np.load(actual_file_path)
    x_train = data["x_train"]  # Shape: (n_samples, 12, 3)
    y_train = data["y_train"]  # Shape: (n_samples, 192, 3)
    
    print(f"Loaded training data:")
    print(f"  Input shape: {x_train.shape} (samples, actuator_tokens, features)")
    print(f"  Output shape: {y_train.shape} (samples, position_tokens, force_xyz)")
    
    return x_train, y_train


# ============================================================
# VISUALIZATION FUNCTIONS  
# ============================================================

def plot_random_sample(x_train, y_train, sample_idx=None):
    """
    Plot a random data sample showing actuator states and distributed forces
    
    Args:
        x_train: (n_samples, 12, 3) - actuator tokens [initial, deformed, difference]
        y_train: (n_samples, n_positions, 3) - force tokens [fx, fy, fz]
        sample_idx: specific sample index, if None uses random
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if sample_idx is None:
        sample_idx = np.random.randint(0, x_train.shape[0])
    
    # Get the sample data
    actuators = x_train[sample_idx]  # (12, 3) - [initial, deformed, difference]
    forces = y_train[sample_idx]     # (n_positions, 3) - [fx, fy, fz]
    
    n_positions = forces.shape[0]
    
    print(f"Plotting sample {sample_idx}:")
    print(f"Actuator states (12 actuators):")
    for i in range(12):
        initial, deformed, diff = actuators[i]
        print(f"  Actuator {i+1}: Initial={initial:.3f}, Deformed={deformed:.3f}, Diff={diff:.3f}")
    
    # Create robot backbone geometry
    # Assume 4 segments of equal length, total length = 16
    L_total = 16.0
    s_positions = np.linspace(0, L_total, n_positions)
    
    # Create a simple curved backbone based on actuator deformations
    # Use actuator differences to create curvature
    backbone_x = np.zeros(n_positions)
    backbone_y = np.zeros(n_positions)
    backbone_z = s_positions.copy()
    
    # Apply simple bending based on actuator states
    segments_per_section = n_positions // 4
    for seg in range(4):
        start_idx = seg * segments_per_section
        end_idx = min((seg + 1) * segments_per_section, n_positions)
        
        if seg < 4 and seg * 3 + 2 < len(actuators):
            # Get actuator differences for this segment (L1, L2, L3)
            l1_diff = actuators[seg * 3][2]     # difference for L1
            l2_diff = actuators[seg * 3 + 1][2] # difference for L2  
            l3_diff = actuators[seg * 3 + 2][2] # difference for L3
            
            # Create simple bending based on actuator differences
            bend_x = (l2_diff - l1_diff) * 0.5  # Differential bending
            bend_y = l3_diff * 0.5              # Extension/compression
            
            # Apply curvature to this segment
            for i in range(start_idx, end_idx):
                progress = (i - start_idx) / max(1, end_idx - start_idx - 1)
                backbone_x[i] = bend_x * progress * (s_positions[i] / L_total)
                backbone_y[i] = bend_y * progress * (s_positions[i] / L_total)
    
    # Create the plot
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: 3D robot shape with forces
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot backbone
    ax1.plot(backbone_x, backbone_y, backbone_z, 'k-', linewidth=3, label='Backbone')
    
    # Plot distributed force arrows
    max_force = np.max(np.linalg.norm(forces, axis=1)) + 1e-12
    scale_factor = 2.0 / max_force
    
    # Sample every nth force vector for clarity
    step_size = max(1, n_positions // 15)
    for i in range(0, n_positions, step_size):
        pos_x, pos_y, pos_z = backbone_x[i], backbone_y[i], backbone_z[i]
        fx, fy, fz = forces[i]
        
        # Scale and plot force arrow
        arrow_fx = scale_factor * fx
        arrow_fy = scale_factor * fy
        arrow_fz = scale_factor * fz
        
        ax1.quiver(pos_x, pos_y, pos_z,
                  arrow_fx, arrow_fy, arrow_fz,
                  color='red', alpha=0.7, linewidth=1.5)
    
    # Show actuator positions as colored spheres
    segment_positions = [0, L_total/4, L_total/2, 3*L_total/4]
    colors = ['blue', 'green', 'orange', 'purple']
    
    for seg in range(4):
        if seg < len(segment_positions):
            z_pos = segment_positions[seg]
            seg_idx = int(seg * n_positions / 4)
            x_pos = backbone_x[seg_idx] if seg_idx < len(backbone_x) else 0
            y_pos = backbone_y[seg_idx] if seg_idx < len(backbone_y) else 0
            
            # Show all 3 actuators for this segment
            for act in range(3):
                act_idx = seg * 3 + act
                if act_idx < len(actuators):
                    _, _, diff = actuators[act_idx]
                    sphere_size = 50 + 200 * abs(diff) / (np.max(np.abs(actuators[:, 2])) + 1e-12)
                    offset = (act - 1) * 0.3  # Spread actuators slightly
                    ax1.scatter([x_pos + offset], [y_pos], [z_pos],
                               c=[colors[seg]], s=sphere_size, alpha=0.8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z (Arc Length)')
    ax1.set_title(f'3D Robot Shape - Sample {sample_idx}')
    ax1.grid(True)
    
    # Plot 2: Force magnitude along backbone
    ax2 = fig.add_subplot(122)
    force_magnitudes = np.linalg.norm(forces, axis=1)
    ax2.plot(s_positions, force_magnitudes, 'b-', linewidth=2, label='Force Magnitude')
    ax2.plot(s_positions, forces[:, 0], 'r--', alpha=0.7, label='Fx')
    ax2.plot(s_positions, forces[:, 1], 'g--', alpha=0.7, label='Fy') 
    ax2.plot(s_positions, forces[:, 2], 'm--', alpha=0.7, label='Fz')
    ax2.set_xlabel('Arc Length')
    ax2.set_ylabel('Force')
    ax2.set_title('Force Distribution Along Backbone')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nForce Statistics:")
    print(f"  Max force magnitude: {np.max(force_magnitudes):.6f}")
    print(f"  Mean force magnitude: {np.mean(force_magnitudes):.6f}")
    print(f"  Force ranges: Fx=[{forces[:,0].min():.6f}, {forces[:,0].max():.6f}]")
    print(f"               Fy=[{forces[:,1].min():.6f}, {forces[:,1].max():.6f}]")
    print(f"               Fz=[{forces[:,2].min():.6f}, {forces[:,2].max():.6f}]")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Load training data
    print("Loading training data...")
    x_train, y_train = load_training_data("11_30_training.npz")
    
    # Plot a random sample to verify data
    print("\nPlotting random sample to verify data quality...")
    plot_random_sample(x_train, y_train)

