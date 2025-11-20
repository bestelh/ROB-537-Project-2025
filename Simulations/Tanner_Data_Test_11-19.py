

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

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
# Main function: bending_3D
# ---------------------------
def bending_3D(EI, GJ, r, N, m, L1, L2, L3, Base_coord,
               Fx, Fy, Fz, Mx, My, Mz, s_force, f_dist_vec, ploton=False):
    """
    Returns (p_def, p_init)
    - L1,L2,L3 expected as sequences length m
    - s_force: array of sample s positions (1D) or None
    - f_dist_vec: either callable f(s)->(3,) OR array shape (3, len(s_force))
    """
    # prepare distributed_force_fun
    if callable(f_dist_vec):
        distributed_force_fun = f_dist_vec
    elif (f_dist_vec is not None) and (s_force is not None):
        s_force = np.asarray(s_force)
        f_dist_arr = np.asarray(f_dist_vec)
        # expected shape (3, n_points)
        if f_dist_arr.shape[0] != 3 and f_dist_arr.shape[1] == 3:
            # maybe given transposed -> fix
            f_dist_arr = f_dist_arr.T
        # create interpolator along axis=1 (columns = points)
        interp = interp1d(s_force, f_dist_arr, kind='linear', axis=1,
                          fill_value='extrapolate', assume_sorted=True)
        def distributed_force_fun(s):
            """Return shape (3,) for scalar s or (3,n) for array s"""
            s_arr = np.atleast_1d(s)
            vals = interp(s_arr)  # shape (3, n)
            if np.isscalar(s):
                return vals[:,0]
            return vals
    else:
        distributed_force_fun = lambda s: np.array([0.0, 0.0, 0.0])

    # Initialize kappa0_vals and theta_init
    kappa0_vals = np.zeros((3, m*N))
    theta_init = np.zeros((m, N))

    # Fill curvature per segment (like MATLAB loop)
    s_vals_accum = np.zeros(m*N)
    max_s = 0.0
    for i in range(m):
        T, curvature, Lck_i, beta, theta = gen_transform_2(L1[i], L2[i], L3[i], r, Base_coord)
        s_local = np.linspace(0.0, Lck_i, N) + max_s
        s_vals_accum[i*N:(i+1)*N] = s_local
        max_s = s_local.max()
        theta_init[i, :] = theta * np.ones(N)
        curvature_magnitude = beta / Lck_i if Lck_i != 0 else 0.0
        kappa0_vals[0, i*N:(i+1)*N] = curvature_magnitude * np.cos(theta_init[i,:] - np.pi/2.0)
        kappa0_vals[1, i*N:(i+1)*N] = curvature_magnitude * np.sin(theta_init[i,:] - np.pi/2.0)
        # z component already zero

    L_total = np.sum(np.array(L1) + np.array(L2) + np.array(L3)) / 3.0
    s_vals = np.linspace(0.0, L_total, m*N)

    # kappa0_fun similar to MATLAB interp1 (linear)
    kappa_interp = interp1d(s_vals, kappa0_vals, kind='linear', axis=1,
                            fill_value='extrapolate', assume_sorted=True)
    def kappa0_fun(s):
        s_arr = np.atleast_1d(s)
        vals = kappa_interp(s_arr)
        if np.isscalar(s):
            return vals[:,0]
        return vals

    # External loads
    Fext = np.array([Fx, Fy, Fz])
    Mext = np.array([Mx, My, Mz])

    # Boundary / params
    p_base = np.array([0.0, 0.0, 0.0])
    R_base = np.eye(3)
    n_tip = Fext
    m_tip = Mext
    Kb = EI * np.eye(3)
    Kt = GJ

    params = {
        'distributed_force_fun': distributed_force_fun,
        'Kb': Kb,
        'Kt': Kt,
        'Fext': Fext,
        'Mext': Mext,
        'p_base': p_base,
        'R_base': R_base,
        'kappa0_fun': kappa0_fun,
        'n_tip': n_tip,
        'm_tip': m_tip
    }

    # Initial guess for solver
    y_init = cosserat_rod_guess(s_vals, params)  # shape (18, m*N)

    # Solve BVP
    sol = solve_bvp(lambda s, y: cosserat_rod_ode(s, y, params),
                    lambda ya, yb: cosserat_rod_bc(ya, yb, params),
                    s_vals, y_init, tol=1e-4, max_nodes=5000)

    if not sol.success:
        print("Warning: solve_bvp did not converge:", sol.message)

    y_sol = sol.sol(s_vals)  # (18, len(s_vals))
    p_def = y_sol[0:3, :]

    if ploton:
        # compute initial (zero-force) shape using distributed_force_fun = zero
        params2 = params.copy()
        params2['distributed_force_fun'] = lambda s: np.array([0.0,0.0,0.0])
        params2['Fext'] = np.array([0.0,0.0,0.0])
        params2['Mext'] = np.array([0.0,0.0,0.0])
        params2['n_tip'] = np.array([0.0,0.0,0.0])
        params2['m_tip'] = np.array([0.0,0.0,0.0])

        y_init2 = cosserat_rod_guess(s_vals, params2)
        sol2 = solve_bvp(lambda s, y: cosserat_rod_ode(s, y, params2),
                         lambda ya, yb: cosserat_rod_bc(ya, yb, params2),
                         s_vals, y_init2, tol=1e-4, max_nodes=5000)
        y_sol2 = sol2.sol(s_vals)
        p_init = y_sol2[0:3, :]

        # Plot deformed and initial
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(p_def[0,:], p_def[1,:], p_def[2,:], 'b-', linewidth=2, label='Deformed Shape')
        ax.plot(p_init[0,:], p_init[1,:], p_init[2,:], 'r-', linewidth=2, label='Initial Shape')

        # Applied tip force arrow
        Fmag = np.linalg.norm([Fx, Fy, Fz]) + 1e-12
        ax.quiver(p_def[0,-1], p_def[1,-1], p_def[2,-1],
                  Fx/Fmag, Fy/Fmag, Fz/Fmag,
                  length=1.0, color='g', linewidth=2, label='Applied Force')

        # Plot distributed force arrows along rod
        num_force_vectors = m*N
        idxs = np.round(np.linspace(0, len(s_vals)-1, num_force_vectors)).astype(int)

        # Evaluate force magnitudes
        f_mags = np.zeros(len(idxs))
        for j, idx in enumerate(idxs):
            sj = s_vals[idx]
            f_vec = distributed_force_fun(sj)
            f_mags[j] = np.linalg.norm(f_vec)

        max_arrow_length = 2.0
        max_force_mag = f_mags.max() + 1e-12
        scale_factor = max_arrow_length / max_force_mag

        for j, idx in enumerate(idxs):
            pos = p_def[:, idx]
            sj = s_vals[idx]
            f_vec = distributed_force_fun(sj)
            f_mag = np.linalg.norm(f_vec)
            if f_mag < 1e-12:
                continue
            f_dir = f_vec / f_mag
            arrow_length = scale_factor * f_mag
            ax.quiver(pos[0], pos[1], pos[2],
                      arrow_length*f_dir[0], arrow_length*f_dir[1], arrow_length*f_dir[2],
                      linewidth=1.5)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.grid(True); ax.set_box_aspect([1,1,1])
        ax.set_xlim([-12,12]); ax.set_ylim([-12,12]); ax.set_zlim([0,16])
        ax.set_title('3D Backbone Shape from solve_bvp')
        ax.legend()
        plt.show()
    else:
        p_init = None

    return y_sol,p_def, p_init

def compute_actuator_lengths_from_solution_3D_full(y_sol, s_vals, m, N, r, L1, L2, L3):
    L1_def = np.zeros(m)
    L2_def = np.zeros(m)
    L3_def = np.zeros(m)

    phis = [0, 2*np.pi/3, 4*np.pi/3]

    # Expected segment lengths
    segment_lengths = (L1 + L2 + L3) / 3.0
    segment_cumsum = np.cumsum(segment_lengths)
    segment_starts = np.insert(segment_cumsum[:-1], 0, 0)

    for i in range(m):
        # Find indices in s_vals corresponding to segment start/end (closest match)
        idx_start = np.searchsorted(s_vals, segment_starts[i], side='left')
        idx_end = np.searchsorted(s_vals, segment_cumsum[i], side='right')
        
        # slice solution accordingly
        p_seg = y_sol[0:3, idx_start:idx_end]
        R_seg_flat = y_sol[3:12, idx_start:idx_end]

        N_seg = p_seg.shape[1]  # number of points in segment slice

        R_seg = np.empty((3,3,N_seg))
        for j in range(N_seg):
            R_seg[:,:,j] = R_seg_flat[:,j].reshape((3,3), order='F')

        act_pts = []
        for phi in phis:
            offset_local = np.array([r * np.cos(phi), r * np.sin(phi), 0])
            pts = np.empty((3, N_seg))
            for j in range(N_seg):
                pts[:, j] = p_seg[:, j] + R_seg[:, :, j] @ offset_local
            act_pts.append(pts)

        def arc_length(pts):
            diffs = np.diff(pts, axis=1)
            seg_lengths = np.linalg.norm(diffs, axis=0)
            return np.sum(seg_lengths)

        L1_def[i] = arc_length(act_pts[0])
        L2_def[i] = arc_length(act_pts[1])
        L3_def[i] = arc_length(act_pts[2])

    return L1_def, L2_def, L3_def



# Actuator lengths
L1 = np.array([4, 5, 5, 4])
L2 = np.array([5, 5, 5, 5])
L3 = np.array([5, 5, 5, 5])

# Average combined segment length
Lck = np.sum((L1 + L2 + L3)) / 3

# Toggle plotting
ploton = False

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

#BEGIN

# ---------- Helpers ----------
def profile_from_control_points(s, num_ctrl, ctrl_vals):
    """
    Build a smooth profile on s from control point values ctrl_vals (len=num_ctrl).
    Interpolates with cubic spline between control points placed uniformly on s range.
    """
    s = np.asarray(s)
    num_ctrl = len(ctrl_vals)
    ctrl_pos = np.linspace(s[0], s[-1], num_ctrl)
    interp = interp1d(ctrl_pos, ctrl_vals, kind='cubic', fill_value='extrapolate', assume_sorted=True)
    return interp(s)


def profile_piecewise_blocks(s, num_blocks, block_vals):
    #TODO: This one is having issues, take it out later
    s = np.asarray(s)
    N = len(s)
    edges = np.linspace(0, N, num_blocks+1, dtype=int)
    prof = np.zeros_like(s)
    for i in range(num_blocks):
        prof[edges[i]:edges[i+1]] = block_vals[i]
    return prof

def profile_filtered_noise(s, scale, sigma):
    
    N = len(s)
    noise = np.random.normal(scale=scale, size=N)
    # gaussian filter -> low-pass
    return gaussian_filter1d(noise, sigma=max(1.0, sigma))

# ---------- Samplers ----------
def sample_spline(s, amp, num_ctrl=6):
    """Return smooth profile from random control points in [-amp, amp]."""
    ctrl_vals = np.random.uniform(-amp, amp, size=num_ctrl)
    return profile_from_control_points(s, num_ctrl, ctrl_vals)

def sample_gaussians(s, amp, min_bumps=1, max_bumps=5,
                     min_width_frac=0.02, max_width_frac=0.25):
    
    """
    Generate a random sum of Gaussians:
    - random number of bumps
    - random amplitudes (sign + magnitude)
    - random centers
    - random widths (log-uniform distribution)
    """

    s = np.asarray(s)
    L = s[-1] - s[0]

    # Number of Gaussian bumps
    k = np.random.randint(min_bumps, max_bumps + 1)

    # Random amplitudes in [-1, 1]
    raw_amps = np.random.uniform(-1.0, 1.0, size=k)

    # Random centers along s
    centers = np.random.uniform(s[0], s[-1], size=k)

    # Log-uniform sampling of widths  (better than linear)
    min_w = L * min_width_frac
    max_w = L * max_width_frac
    widths = np.exp(np.random.uniform(np.log(min_w), np.log(max_w), size=k))

    # Build the Gaussian sum
    prof = np.zeros_like(s)
    for a, c, w in zip(raw_amps, centers, widths):
        prof += a * np.exp(-0.5 * ((s - c) / w)**2)

    # Scale to desired amplitude
    return amp * prof

def sample_fourier(s, amp, min_modes=1, max_modes=8):
    L = s[-1] - s[0]

    # Random number of Fourier modes
    n_modes = np.random.randint(min_modes, max_modes + 1)

    # Random amplitudes for each mode (raw, not scaled)
    amps = np.random.uniform(-1.0, 1.0, size=n_modes)

    # Build the Fourier sum
    prof = np.zeros_like(s)
    for k, a in enumerate(amps, start=1):
        prof += a * np.sin(2 * np.pi * k * (s - s[0]) / L)

    # Scale to [-amp, +amp]
    return amp * prof

def sample_piecewise(s, amp, num_blocks=8):
    block_vals = np.random.uniform(-amp, amp, size=num_blocks)
    return profile_piecewise_blocks(s, num_blocks, block_vals)

def sample_filtered_noise(s, amp, sigma_fraction=0.02):
    L = s[-1] - s[0]
    sigma = max(1.0, int(len(s) * sigma_fraction))
    return profile_filtered_noise(s, amp, sigma)


# ---------- Single interface ----------
def generate_random_force_profile(s, amp, method='spline', **kwargs):
    """
    s: 1D array of arc positions
    amp: scalar amplitude (max absolute force)
    method: one of 'spline','gaussian','fourier','blocks','noise'
    kwargs: method-specific params
    """
    if method == 'spline':
        return sample_spline(s, amp, num_ctrl=kwargs.get('num_ctrl', 8))
    elif method == 'gaussian':
        return sample_gaussians(s, amp, max_bumps=kwargs.get('max_bumps', 4))
    elif method == 'fourier':
        return sample_fourier(s, amp)
    elif method == 'blocks':
        return sample_piecewise(s, amp, num_blocks=kwargs.get('num_blocks', 8))
    elif method == 'noise':
        return sample_filtered_noise(s, amp, sigma_fraction=kwargs.get('sigma_fraction', 0.02))
    else:
        raise ValueError("Unknown method")

#END

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


# Constants and fixed parameters
EI = 10.0        # Bending stiffness
GJ = 0.7
r = 5 / np.pi    # Robot radius
N = 48          # Points per segment prolly want to mess with this
m = 4            # Number of segments

# Initial actuator positions
act_11_start = np.array([r, 0, 0, 1])
act_12_start = np.array([r * np.cos(2 * np.pi / 3), r * np.sin(2 * np.pi / 3), 0, 1])
act_13_start = np.array([r * np.cos(4 * np.pi / 3), r * np.sin(4 * np.pi / 3), 0, 1])
Base_coord = np.eye(4)

# Forces applied at tip (N)
Fx, Fy, Fz = 0.0, 0.0, 0.0

# Moments applied at tip
Mx, My, Mz = 0.0, 0.0, 0.0

# Number of segments
m = 4

# Fixed values for base coordinate and external tip forces/moments
Base_coord = np.eye(4)
Fx, Fy, Fz = 0.0, 0.0, 0.0
Mx, My, Mz = 0.0, 0.0, 0.0

# Sampling ranges
length_min, length_max = 4.0, 8.0
force_min, force_max = 0.0, 0.04

# Number of random samples to run
num_samples = 10000  # Adjust for compute budget


def single_simulation(params):
    np.random.seed(None)  # re-seed from system entropy

    L1, L2, L3, f_amp_x, f_amp_y, f_amp_z = params

    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)

    Lck = np.sum((L1 + L2 + L3)) / 3.0
    s_force = np.linspace(0, Lck, m * N)


    # ----- Choose one profile type -----
    method = np.random.choice(
        ['spline', 'gaussian', 'fourier', 'noise'],
        p=[0.25, 0.50, 0.20, 0.05]
    )

    # ----- Generate one base profile shape (range: roughly [-1, 1]) -----
    base_profile = generate_random_force_profile(s_force, amp=1.0, method=method)

    # ----- Normalize shape so max |value| = 1 -----
    mx = np.max(np.abs(base_profile))
    if mx > 0:
        base_profile = base_profile / mx

    # ----- Apply different amplitudes on each axis -----
    fx_profile = f_amp_x * base_profile 
    fy_profile = f_amp_y * base_profile 
    fz_profile = f_amp_z * base_profile 

    # stack into 3xN array
    f_dist_data = np.vstack([fx_profile, fy_profile, fz_profile])

    try:
        y_sol, p_def, p_init = bending_3D(
            EI, GJ, r, N, m, L1, L2, L3, Base_coord,
            Fx, Fy, Fz, Mx, My, Mz, s_force, f_dist_data, False)

        L1_def, L2_def, L3_def = compute_actuator_lengths_from_solution_3D_full(
            y_sol, s_force, m, N, r, L1, L2, L3)

        # Prepare result summary
        result = {
            'force_amp_x': f_amp_x,
            'force_amp_y': f_amp_y,
            'force_amp_z': f_amp_z,
            'method': method,
            
        }

        for i in range(m):
            result[f'L1_input_{i+1}'] = L1[i]
            result[f'L2_input_{i+1}'] = L2[i]
            result[f'L3_input_{i+1}'] = L3[i]
            result[f'L1_def_{i+1}'] = L1_def[i]
            result[f'L2_def_{i+1}'] = L2_def[i]
            result[f'L3_def_{i+1}'] = L3_def[i]

        # Add full force distributions (saved later in a .npz)
        result['f_dist_fx'] = fx_profile
        result['f_dist_fy'] = fy_profile
        result['f_dist_fz'] = fz_profile
        result['s_force'] = s_force

        return result

    except Exception as e:
        print(f"Failed simulation with params {params}: {e}")
        return None


def generate_random_params(num_samples):
    params_list = []
    for _ in range(num_samples):
        L1 = np.random.uniform(length_min, length_max, size=m)
        L2 = np.random.uniform(length_min, length_max, size=m)
        L3 = np.random.uniform(length_min, length_max, size=m)
        f_amp_x = np.random.uniform(force_min, force_max) * np.random.choice([-1, 1])
        f_amp_y = np.random.uniform(force_min, force_max) * np.random.choice([-1, 1])
        f_amp_z = np.random.uniform(force_min, force_max) * np.random.choice([-1, 1])
        params_list.append((L1, L2, L3, f_amp_x, f_amp_y, f_amp_z))
    return params_list



from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    params_list = generate_random_params(num_samples)
    results = []
    max_workers = 10

    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(single_simulation, p) for p in params_list]

    try:
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"Simulation failed: {e}")

            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_samples} simulations")

    except KeyboardInterrupt:
        print("\nCtrl-C detected! Attempting graceful shutdown...")

        # Cancel all pending futures to avoid abrupt termination
        for f in futures:
            f.cancel()

        # Save partial results
        np.savez_compressed('simulation_results_partial_11-19.npz', data=results)
        print(f"Saved {len(results)} partial results to 'simulation_results_partial.npz'")

    finally:
        executor.shutdown(wait=False)
        print("Executor shutdown complete.")

    # Save full results if finished normally
    if len(results) == num_samples:
        np.savez_compressed('simulation_results_all_11-19.npz', data=results)
        print(f"Saved all {len(results)} results to 'simulation_results_all.npz'")

if __name__ == "__main__":
    main()

