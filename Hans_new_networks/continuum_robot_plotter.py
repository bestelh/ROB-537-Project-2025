import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ---------------------------
# Helper functions
# ---------------------------
def skew(v):
    """Return skew-symmetric matrix for vector v (3,)"""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])

def gen_transform_2(L_k1, L_k2, L_k3, r, T_prev):
    """
    Returns (T_k, rho_k, L_ck, beta_k, theta_k)
    T_prev is unused here (kept for compatibility)
    """
    L_ck = (L_k1 + L_k2 + L_k3) / 3.0
    # prevent divide-by-zero later
    if L_ck == 0:
        return np.eye(4), 0.0, 0.0, 0.0, 0.0

    beta_k = 2.0 * np.sqrt(L_k1**2 + L_k2**2 + L_k3**2 - L_k1*L_k2 - L_k1*L_k3 - L_k2*L_k3) / (3.0 * r)
    theta_k = np.arctan2(3*(L_k2 - L_k3), np.sqrt(3)*(L_k2 + L_k3 - 2*L_k1))
    rho_k = beta_k / L_ck if abs(L_ck) > 1e-12 else 0.0

    ct = np.cos(theta_k)
    st = np.sin(theta_k)
    cb = np.cos(beta_k)
    sb = np.sin(beta_k)

    R_k = np.array([
        [cb*ct**2 + st**2,      (cb-1)*ct*st,       ct*sb],
        [(cb-1)*ct*st,          ct**2 + cb*st**2,   st*sb],
        [-ct*sb,                -st*sb,             cb]
    ])

    if abs(rho_k) < 1e-6:
        # near-straight transform
        T_k = np.eye(4)
        T_k[2,3] = L_ck
    else:
        P_k = (1.0/rho_k) * np.array([(1-cb)*ct, (1-cb)*st, sb])
        T_k = np.eye(4)
        T_k[:3,:3] = R_k
        T_k[:3,3] = P_k

    return T_k, rho_k, L_ck, beta_k, theta_k

# ---------------------------
# Core ODEs, BCs, initial guess
# ---------------------------
def cosserat_rod_ode(s, y, params):
    """
    s: scalar or array-like (solve_bvp passes array s)
    y: shape (18, len(s))
    returns dy/ds with same shape.
    """
    # Unpack params
    Kb = params['Kb']            # 3x3
    distributed_force_fun = params['distributed_force_fun']
    kappa0_fun = params['kappa0_fun']

    # prepare output
    num_s = s.size
    dy = np.zeros_like(y)

    for idx in range(num_s):
        si = s[idx]
        yi = y[:, idx]

        # unpack state
        p = yi[0:3]
        R = yi[3:12].reshape((3,3), order='F')   # column-major like MATLAB
        n = yi[12:15]
        m_vec = yi[15:18]

        t = R[:, 2]  # third column

        # curvature vectors
        kappa0 = kappa0_fun(si)            # should return (3,) vector
        # Kb \ (m - Kb*kappa0) in MATLAB => np.linalg.solve(Kb, ..)
        kappa = np.linalg.solve(Kb, m_vec - Kb.dot(kappa0))

        dp_ds = t
        dR_ds = R.dot(skew(kappa))

        # distributed force at si
        f_dist = distributed_force_fun(si)
        # ensure shape (3,)
        f_dist = np.asarray(f_dist).flatten()

        dn_ds = -f_dist
        dm_ds = -np.cross(t, n)

        dy[0:3, idx] = dp_ds
        # flatten dR_ds into 9x1 column by column (MATLAB order)
        dy[3:12, idx] = dR_ds.reshape(9, order='F')
        dy[12:15, idx] = dn_ds
        dy[15:18, idx] = dm_ds

    return dy

def cosserat_rod_bc(ya, yb, params):
    """
    Boundary conditions residuals: returns 18-length vector
    ya: y at s=0, yb: y at s=L
    """
    p_base = params['p_base']
    R_base = params['R_base']
    n_tip = params['n_tip']
    m_tip = params['m_tip']

    res = np.zeros(18)
    res[0:3] = ya[0:3] - p_base
    res[3:12] = ya[3:12] - R_base.reshape(9, order='F')
    res[12:15] = yb[12:15] - n_tip
    res[15:18] = yb[15:18] - m_tip
    return res

def cosserat_rod_guess(s, params):
    """
    s: array of s points -> returns yguess with shape (18, s.size)
    """
    s = np.atleast_1d(s)
    out = np.zeros((18, s.size))
    kappa0_fun = params['kappa0_fun']
    
    n_tip = params['n_tip']
    m_tip = params['m_tip']

    for i, si in enumerate(s):
        kappa0 = kappa0_fun(si)
        beta_val = np.linalg.norm(kappa0) * si
        if beta_val < 1e-6:
            tangent = np.array([0.0, 0.0, 1.0])
        else:
            theta_val = np.arctan2(kappa0[1], kappa0[0])
            tangent = np.array([np.sin(beta_val)*np.cos(theta_val),
                                np.sin(beta_val)*np.sin(theta_val),
                                np.cos(beta_val)])
        p = tangent * si
        R = np.eye(3)
        yinit = np.concatenate([p, R.reshape(9, order='F'), n_tip, m_tip])
        out[:, i] = yinit
    return out

# ---------------------------
# Simple plotting function
# ---------------------------
def plot_robot_shapes(p_init, p_def, s_force, f_dist_data):
    """
    Simple plotting function for continuum robot visualization
    
    Args:
        p_init: Initial shape positions (3, N) array [x, y, z]
        p_def: Deformed shape positions (3, N) array [x, y, z]  
        s_force: Arc length coordinates (N,) array
        f_dist_data: Force distribution (3, N) or (N, 3) array [fx, fy, fz]
    """
    
    # Ensure arrays and fix shapes if needed
    p_init = np.asarray(p_init)
    p_def = np.asarray(p_def) 
    s_force = np.asarray(s_force)
    f_dist_data = np.asarray(f_dist_data)
    
    # Fix force data shape if needed (should be 3 x N)
    if f_dist_data.shape[0] != 3 and f_dist_data.shape[1] == 3:
        f_dist_data = f_dist_data.T
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot deformed and initial shapes
    ax.plot(p_def[0,:], p_def[1,:], p_def[2,:], 'b-', linewidth=2, label='Deformed Shape')
    ax.plot(p_init[0,:], p_init[1,:], p_init[2,:], 'r-', linewidth=2, label='Initial Shape')

    # Plot distributed force arrows along rod
    n_points = p_def.shape[1]
    step_size = max(1, n_points // 20)  # Show about 20 arrows
    
    # Calculate force magnitudes for scaling
    force_mags = np.linalg.norm(f_dist_data, axis=0)
    max_force_mag = np.max(force_mags) + 1e-12
    max_arrow_length = 2.0
    scale_factor = max_arrow_length / max_force_mag

    # Plot force arrows
    for i in range(0, n_points, step_size):
        pos = p_def[:, i]
        f_vec = f_dist_data[:, i]
        f_mag = np.linalg.norm(f_vec)
        
        if f_mag < 1e-12:
            continue
            
        # Scale and plot force arrow
        f_dir = f_vec / f_mag
        arrow_length = scale_factor * f_mag
        ax.quiver(pos[0], pos[1], pos[2],
                  arrow_length*f_dir[0], arrow_length*f_dir[1], arrow_length*f_dir[2],
                  color='green', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.grid(True)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-12,12])
    ax.set_ylim([-12,12])
    ax.set_zlim([0,16])
    ax.set_title('Continuum Robot: Initial vs Deformed Shape with Distributed Forces')
    ax.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example data - you would replace this with your actual data
    
    # Example initial and deformed shapes (3 x N arrays)
    s = np.linspace(0, 16, 64)
    # Initial shape (straight)
    p_init = np.array([
        np.zeros(64),  # x
        np.zeros(64),  # y  
        s              # z
    ])
    
    # Deformed shape (curved)
    p_def = np.array([
        0.5 * np.sin(np.pi * s / 16),  # x - some bending
        0.2 * s / 16,                   # y - slight offset
        s                               # z - same length
    ])
    
    # Example force distribution (3 x N array)
    fx = 0.05 * np.sin(2 * np.pi * s / 16)
    fy = 0.03 * np.cos(np.pi * s / 8)
    fz = 0.02 * np.ones_like(s)
    f_dist_data = np.array([fx, fy, fz])
    
    # Plot the robot
    plot_robot_shapes(p_init, p_def, s, f_dist_data)