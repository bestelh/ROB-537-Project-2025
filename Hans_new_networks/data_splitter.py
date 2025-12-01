import numpy as np
from pathlib import Path

def data_import(npz_file_path="simulation_results_partial_11-30.npz", train_split=0.5):
    """
    Import and prepare data from the .npz simulation results file.
    
    DATASET USAGE EXPLANATION:
    - Total dataset: ~10,000 simulation results
    - train_split=0.5 means 50% for training, 50% for testing
    - Training data: ~5,000 samples (used to learn network weights)
    - Test data: ~5,000 samples (used to evaluate performance, NEVER seen during training)
    
    Parameters:
    npz_file_path (str): Path to the .npz file containing simulation results
    train_split (float): Fraction of data to use for training (default: 0.5 = 50%)
    
    Returns:
    tuple: (x_train, y_train, x_test, y_test, s_force, normalization_params)
           - x_train: Training inputs (24 features × n_train_samples)
           - y_train: Training targets (576 force values × n_train_samples)
           - x_test: Test inputs (24 features × n_test_samples)
           - y_test: Test targets (576 force values × n_test_samples)
           - s_force: Arc length positions for force profiles
           - normalization_params: Statistics for denormalizing predictions
    """
    print(f"Loading data from {npz_file_path}...")
    
    # Try multiple possible file locations
    script_dir = Path(__file__).parent
    possible_files = [
        npz_file_path,
        script_dir / npz_file_path,
        script_dir.parent / npz_file_path,  # Look in parent Simulations folder
        Path.cwd() / npz_file_path,
        Path.cwd() / "Simulations" / npz_file_path
    ]
    
    actual_file_path = None
    for filepath in possible_files:
        if Path(filepath).exists():
            actual_file_path = str(filepath)
            break
    
    if actual_file_path is None:
        raise FileNotFoundError(f"Could not find {npz_file_path} in any of the searched locations")
    
    # Load the data
    npz_data = np.load(actual_file_path, allow_pickle=True)
    simulation_results = npz_data['data']
    
    print(f"Total number of simulations: {len(simulation_results)}")
    
    # Extract features and targets
    input_features = []
    output_forces = []
    
    for sim in simulation_results:
        # Input: Each actuator as a token with 3 features [initial, deformed, difference]
        # 12 tokens total (L1,L2,L3 for each of 4 segments)
        tokens = []
        
        # Segment 1
        tokens.append([sim['L1_input_1'], sim['L1_def_1'], sim['L1_def_1'] - sim['L1_input_1']])
        tokens.append([sim['L2_input_1'], sim['L2_def_1'], sim['L2_def_1'] - sim['L2_input_1']])
        tokens.append([sim['L3_input_1'], sim['L3_def_1'], sim['L3_def_1'] - sim['L3_input_1']])
        
        # Segment 2
        tokens.append([sim['L1_input_2'], sim['L1_def_2'], sim['L1_def_2'] - sim['L1_input_2']])
        tokens.append([sim['L2_input_2'], sim['L2_def_2'], sim['L2_def_2'] - sim['L2_input_2']])
        tokens.append([sim['L3_input_2'], sim['L3_def_2'], sim['L3_def_2'] - sim['L3_input_2']])
        
        # Segment 3
        tokens.append([sim['L1_input_3'], sim['L1_def_3'], sim['L1_def_3'] - sim['L1_input_3']])
        tokens.append([sim['L2_input_3'], sim['L2_def_3'], sim['L2_def_3'] - sim['L2_input_3']])
        tokens.append([sim['L3_input_3'], sim['L3_def_3'], sim['L3_def_3'] - sim['L3_input_3']])
        
        # Segment 4
        tokens.append([sim['L1_input_4'], sim['L1_def_4'], sim['L1_def_4'] - sim['L1_input_4']])
        tokens.append([sim['L2_input_4'], sim['L2_def_4'], sim['L2_def_4'] - sim['L2_input_4']])
        tokens.append([sim['L3_input_4'], sim['L3_def_4'], sim['L3_def_4'] - sim['L3_input_4']])
        
        # Output: Force tokens with 3 dimensions [fx, fy, fz] at each position
        # 192 positions with 3 force components each
        force_tokens = []
        fx = sim['f_dist_fx']  # 192 values - Force X
        fy = sim['f_dist_fy']  # 192 values - Force Y
        fz = sim['f_dist_fz']  # 192 values - Force Z
        
        for i in range(len(fx)):
            force_tokens.append([fx[i], fy[i], fz[i]])
        
        input_features.append(tokens)
        output_forces.append(force_tokens)
    
    X = np.array(input_features)  # Shape: (n_samples, 12, 3) - 12 tokens, 3 features each
    Y = np.array(output_forces)   # Shape: (n_samples, 192, 3) - 192 positions, 3 forces each
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    
    # Get arc length positions for reference
    s_force = simulation_results[0]['s_force']
    
    # 80/20 split into training and test sets
    # CRITICAL: Test data is COMPLETELY SEPARATED and never used for training!
    split_idx = int(len(X) * 0.8)  # 80% for training
    x_train = X[:split_idx]  # Shape: (n_train, 12, 3) - First 80% for training
    y_train = Y[:split_idx]  # Shape: (n_train, 192, 3)
    x_test = X[split_idx:]   # Shape: (n_test, 12, 3) - Last 20% for testing
    y_test = Y[split_idx:]   # Shape: (n_test, 192, 3)
    
    # Save training data in Hans_new_networks folder
    script_dir = Path(__file__).parent
    training_path = script_dir / "11_30_training.npz"
    np.savez(
        training_path,
        x_train=x_train,
        y_train=y_train,
        s_force=s_force
    )
    print(f"Saved training data to: {training_path}")

    # Save testing data in Hans_new_networks folder
    testing_path = script_dir / "11_30_testing.npz"
    np.savez(
        testing_path,
        x_test=x_test,
        y_test=y_test,
        s_force=s_force
    )
    print(f"Saved testing data to: {testing_path}")
    
    return x_train, y_train, x_test, y_test, s_force


if __name__ == "__main__":
    data_import()
   
   