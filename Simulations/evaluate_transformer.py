import torch
import numpy as np
from pathlib import Path
import sys

# --- Import model and helper functions ---
from Project_NN_Transformer import Sensor2FieldTransformer, compute_whitening

def data_import(npz_file_path):
    npz_data = np.load(npz_file_path, allow_pickle=True)
    simulation_results = npz_data['data']
    input_features = []
    output_forces = []
    for sim in simulation_results:
        inputs = [
            sim['L1_input_1'], sim['L1_def_1']-sim['L1_input_1'], sim['L2_input_1'], sim['L2_def_1']-sim['L2_input_1'], sim['L3_input_1'], sim['L3_def_1']-sim['L3_input_1'],
            sim['L1_input_2'], sim['L1_def_2']-sim['L1_input_2'], sim['L2_input_2'], sim['L2_def_2']-sim['L2_input_2'], sim['L3_input_2'], sim['L3_def_2']-sim['L3_input_2'],
            sim['L1_input_3'], sim['L1_def_3']-sim['L1_input_3'], sim['L2_input_3'], sim['L2_def_3']-sim['L2_input_3'], sim['L3_input_3'], sim['L3_def_3']-sim['L3_input_3'],
            sim['L1_input_4'], sim['L1_def_4']-sim['L1_input_4'], sim['L2_input_4'], sim['L2_def_4']-sim['L2_input_4'], sim['L3_input_4'], sim['L3_def_4']-sim['L3_input_4']
        ]
        forces = np.concatenate([sim['f_dist_fx'], sim['f_dist_fy'], sim['f_dist_fz']])
        input_features.append(inputs)
        output_forces.append(forces)
    X = np.array(input_features)
    Y = np.array(output_forces)
    return X, Y

def load_scalers(scaler_path):
    scalers = np.load(scaler_path, allow_pickle=True).item()
    return scalers

def main():
    # Paths (edit as needed)
    model_path = "transformer_model.pth"
    test_data_path = "simulation_results_all_11-23.npz"
    scaler_path = "transformer_scalers.npz"

    # Load test data
    X_test, Y_test = data_import(test_data_path)

    # Load scalers
    scalers = load_scalers(scaler_path)

    # Build normalized features for transformer
    N = X_test.shape[0]
    X_feats = np.zeros((N, 12, 3), dtype=np.float32)
    for s in range(12):
        before = X_test[:, 2*s].astype(np.float32)
        after  = X_test[:, 2*s+1].astype(np.float32)
        diff = after - before
        mean = scalers["sensor_means"][s].reshape(1,3)
        std  = scalers["sensor_stds"][s].reshape(1,3)
        block = np.stack([before, after, diff], axis=1)
        block_norm = (block - mean) / (std + 1e-12)
        X_feats[:, s, :] = block_norm

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Sensor2FieldTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Run inference
    src_tokens = torch.from_numpy(X_feats).float().to(device)
    with torch.no_grad():
        preds = model.infer(src_tokens)  # (N, 96, 3)
    preds_np = preds.cpu().numpy().reshape(N, -1)  # (N,288)

    # Unwhiten if needed
    if scalers.get("whiten") is not None:
        preds_np = scalers["whiten"]["unwhiten_fn"](preds_np)

    # Save predictions
    np.savez("transformer_predictions.npz", preds=preds_np, targets=Y_test)

    print("Inference complete. Predictions saved to transformer_predictions.npz.")

if __name__ == "__main__":
    main()
