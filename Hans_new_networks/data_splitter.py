import numpy as np
from pathlib import Path

def save_train_test_npz(npz_file_path="simulation_results_all_11-19.npz",
                        output_file="train_test_split.npz"):
    """
    Load simulation results, extract features/forces, split 80/20,
    and save as a .npz file containing:
        x_train, y_train, x_test, y_test, s_force
    """

    print(f"Loading data from {npz_file_path}...")

    # Search for file across common paths
    script_dir = Path(__file__).parent
    possible_files = [
        npz_file_path,
        script_dir / npz_file_path,
        script_dir.parent / npz_file_path,
        Path.cwd() / npz_file_path,
        Path.cwd() / "Simulations" / npz_file_path
    ]

    actual_file_path = None
    for filepath in possible_files:
        if Path(filepath).exists():
            actual_file_path = str(filepath)
            break

    if actual_file_path is None:
        raise FileNotFoundError(f"Could not find {npz_file_path}")

    npz_data = np.load(actual_file_path, allow_pickle=True)
    simulation_results = npz_data["data"]
    print(f"Loaded {len(simulation_results)} simulation entries")

    X_list = []
    Y_list = []

    # -----------------------------
    # Extract features + forces
    # -----------------------------
    for sim in simulation_results:
        inputs = [
            sim['L1_input_1'] - sim['L1_def_1'],
            sim['L2_input_1'] - sim['L2_def_1'],
            sim['L3_input_1'] - sim['L3_def_1'],
            sim['L1_input_2'] - sim['L1_def_2'],
            sim['L2_input_2'] - sim['L2_def_2'],
            sim['L3_input_2'] - sim['L3_def_2'],
            sim['L1_input_3'] - sim['L1_def_3'],
            sim['L2_input_3'] - sim['L2_def_3'],
            sim['L3_input_3'] - sim['L3_def_3'],
            sim['L1_input_4'] - sim['L1_def_4'],
            sim['L2_input_4'] - sim['L2_def_4'],
            sim['L3_input_4'] - sim['L3_def_4']
        ]

        forces = np.concatenate([
            sim["f_dist_fx"],
            sim["f_dist_fy"],
            sim["f_dist_fz"]
        ])

        X_list.append(inputs)
        Y_list.append(forces)

    X = np.array(X_list)
    Y = np.array(Y_list)

    print(f"Raw X shape: {X.shape}")
    print(f"Raw Y shape: {Y.shape}")

    # Outlier filter (same as your original)
    valid_mask = (np.abs(Y).max(axis=1) < 1.0) & (np.abs(X).max(axis=1) < 1000)
    X = X[valid_mask]
    Y = Y[valid_mask]

    print(f"Remaining after filtering: {len(X)} samples")

    # -----------------------------
    # 80/20 split (no shuffle)
    # -----------------------------
    split_idx = int(0.8 * len(X))

    x_train = X[:split_idx].T
    y_train = Y[:split_idx].T
    x_test  = X[split_idx:].T
    y_test  = Y[split_idx:].T

    # Arc length vector
    s_force = simulation_results[0]["s_force"]


    # Save training data
    np.savez(
        "training_data_1.npz",
        x_train=x_train,
        y_train=y_train,
        s_force=s_force
    )
    print("Saved training data to: training_data_1.npz")

    # Save testing data
    np.savez(
        "testing_data_1.npz",
        x_test=x_test,
        y_test=y_test,
        s_force=s_force
    )
    print("Saved testing data to: testing_data_2.npz")

    npz_data.close()
    
if __name__ == "__main__":
    save_train_test_npz()