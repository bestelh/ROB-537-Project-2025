import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import the model classes from transfomer_model_24.py
from transfomer_model_24 import DeformationToForceTransformer, PositionalEncoding

class TransformerModelLoader:
    """Class for loading and testing trained transformer models"""
    
    def __init__(self, model_path='complete_transformer_model_24x3.pth'):
        self.model_path = model_path
        self.model = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = None
        self.val_losses = None
        
    def load_model(self):
        """Load the trained model from saved file"""
        try:
            # Load model data
            model_data = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract configuration
            self.model_config = model_data['model_config']
            self.train_losses = model_data.get('train_losses', [])
            self.val_losses = model_data.get('val_losses', [])
            
            # Create model with saved configuration
            self.model = DeformationToForceTransformer(
                input_dim=self.model_config['input_dim'],
                output_dim=self.model_config['output_dim'],
                output_seq_len=self.model_config.get('output_seq_len', 24),  # Default for backward compatibility
                d_model=self.model_config['d_model'],
                nhead=self.model_config['nhead'],
                num_encoder_layers=self.model_config['num_encoder_layers'],
                num_decoder_layers=self.model_config['num_decoder_layers'],
                dim_feedforward=self.model_config['dim_feedforward'],
                dropout=self.model_config['dropout']
            )
            
            # Load trained weights
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ Model loaded successfully from {self.model_path}")
            print(f"✓ Model configuration: {self.model_config}")
            print(f"✓ Device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def predict(self, actuator_data):
        """
        Make predictions on new actuator data
        
        Args:
            actuator_data: numpy array of shape (n_samples, 12, 3) or (12, 3) for single sample
                          Contains [initial_length, deformed_length, difference] for each actuator
        
        Returns:
            predictions: numpy array of shape (n_samples, 24, 3) or (24, 3) for single sample
                        Contains [fx, fy, fz] for each position along the robot
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Handle single sample input
        single_sample = False
        if actuator_data.ndim == 2:
            actuator_data = actuator_data[np.newaxis, :]  # Add batch dimension
            single_sample = True
        
        # Convert to tensor
        x_tensor = torch.FloatTensor(actuator_data).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(x_tensor)
            predictions = predictions.cpu().numpy()
        
        # Remove batch dimension for single sample
        if single_sample:
            predictions = predictions[0]
        
        return predictions
    
    def find_peak(self, force_curve):
        """Find the peak (point farthest from zero) in a force curve
        
        Args:
            force_curve: 1D array of force values
            
        Returns:
            tuple: (peak_index, peak_value) where peak is farthest from zero
        """
        abs_forces = np.abs(force_curve)
        peak_idx = np.argmax(abs_forces)
        peak_value = force_curve[peak_idx]
        return peak_idx, peak_value
    
    def calculate_peak_differences(self, pred_forces, true_forces):
        """Calculate peak differences for all force components
        
        Args:
            pred_forces: (N, 3) array of predicted forces
            true_forces: (N, 3) array of true forces
            
        Returns:
            dict: Peak analysis results for each component
        """
        components = ['Fx', 'Fy', 'Fz']
        peak_analysis = {}
        
        for i, comp in enumerate(components):
            # Find peaks
            pred_peak_idx, pred_peak_val = self.find_peak(pred_forces[:, i])
            true_peak_idx, true_peak_val = self.find_peak(true_forces[:, i])
            
            # Calculate differences as percentages
            x_diff = abs(pred_peak_idx - true_peak_idx) / 24 * 100  # Position difference as % of total length
            y_diff = abs(pred_peak_val - true_peak_val) / abs(true_peak_val) * 100 if true_peak_val != 0 else 0  # Force difference as % of true peak
            
            peak_analysis[comp] = {
                'pred_peak_idx': pred_peak_idx,
                'pred_peak_val': pred_peak_val,
                'true_peak_idx': true_peak_idx,
                'true_peak_val': true_peak_val,
                'x_diff': x_diff,
                'y_diff': y_diff
            }
            
        return peak_analysis
    
    def evaluate_test_data(self, x_test, y_test=None, show_samples=5):
        """
        Evaluate model on test data
        
        Args:
            x_test: Test actuator data (n_samples, 12, 3)
            y_test: True force data (n_samples, 192, 3) - optional for comparison
            show_samples: Number of samples to visualize
        
        Returns:
            dict: Evaluation metrics and predictions
        """
        print(f"Evaluating on {x_test.shape[0]} test samples...")
        
        # Make predictions
        predictions = self.predict(x_test)
        
        results = {
            'predictions': predictions,
            'input_data': x_test
        }
        
        if y_test is not None:
            # Calculate metrics
            mse = np.mean((predictions - y_test)**2)
            mae = np.mean(np.abs(predictions - y_test))
            max_error = np.max(np.abs(predictions - y_test))
            
            # Per-component metrics
            mse_components = np.mean((predictions - y_test)**2, axis=(0, 1))
            mae_components = np.mean(np.abs(predictions - y_test), axis=(0, 1))
            
            # Peak analysis for all samples
            all_peak_diffs = []
            for i in range(len(predictions)):
                peak_analysis = self.calculate_peak_differences(predictions[i], y_test[i])
                all_peak_diffs.append(peak_analysis)
            
            # Calculate average peak differences
            avg_peak_diffs = {}
            for comp in ['Fx', 'Fy', 'Fz']:
                x_diffs = [pd[comp]['x_diff'] for pd in all_peak_diffs]
                y_diffs = [pd[comp]['y_diff'] for pd in all_peak_diffs]
                avg_peak_diffs[comp] = {
                    'avg_x_diff': np.mean(x_diffs),
                    'avg_y_diff': np.mean(y_diffs),
                    'std_x_diff': np.std(x_diffs),
                    'std_y_diff': np.std(y_diffs)
                }
            
            results.update({
                'true_forces': y_test,
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'mse_fx': mse_components[0],
                'mse_fy': mse_components[1], 
                'mse_fz': mse_components[2],
                'mae_fx': mae_components[0],
                'mae_fy': mae_components[1],
                'mae_fz': mae_components[2],
                'peak_analysis': avg_peak_diffs
            })
            
            print(f"\nEvaluation Results:")
            print(f"  Overall MSE: {mse:.6f}")
            print(f"  Overall MAE: {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")
            print(f"  Component MSE - Fx: {mse_components[0]:.6f}, Fy: {mse_components[1]:.6f}, Fz: {mse_components[2]:.6f}")
            print(f"  Component MAE - Fx: {mae_components[0]:.6f}, Fy: {mae_components[1]:.6f}, Fz: {mae_components[2]:.6f}")
            
            print(f"\nPeak Analysis (Average over all samples):")
            for comp in ['Fx', 'Fy', 'Fz']:
                x_diff = avg_peak_diffs[comp]['avg_x_diff']
                y_diff = avg_peak_diffs[comp]['avg_y_diff']
                print(f"  {comp} - Position diff: {x_diff:.1f}%±{avg_peak_diffs[comp]['std_x_diff']:.1f}%, Force diff: {y_diff:.1f}%±{avg_peak_diffs[comp]['std_y_diff']:.1f}%")
        
        return results
    
    def _visualize_random_sample(self, x_test, y_test):
        """Visualize a single random sample prediction with peak detection"""
        import random
        
        # Select random sample
        sample_idx = random.randint(0, x_test.shape[0] - 1)
        print(f"\nShowing random sample #{sample_idx + 1} from {x_test.shape[0]} test samples")
        
        # Make prediction for this sample
        x_sample = x_test[sample_idx:sample_idx+1]  # Keep batch dimension
        prediction = self.predict(x_sample)
        y_sample = y_test[sample_idx]
        
        # Calculate metrics for this sample
        mse = np.mean((prediction[0] - y_sample)**2)
        mae = np.mean(np.abs(prediction[0] - y_sample))
        
        # Peak analysis for this sample
        peak_analysis = self.calculate_peak_differences(prediction[0], y_sample)
        
        # Print peak analysis for this sample
        print(f"\nPeak Analysis for Sample #{sample_idx + 1}:")
        for comp in ['Fx', 'Fy', 'Fz']:
            x_diff = peak_analysis[comp]['x_diff']
            y_diff = peak_analysis[comp]['y_diff']
            print(f"  {comp}: Position diff = {x_diff:.1f}%, Force diff = {y_diff:.1f}%")
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Random Test Sample #{sample_idx + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}', fontsize=14)
        
        for j, component in enumerate(['Fx', 'Fy', 'Fz']):
            # Top row: Predictions vs True with peaks marked
            axes[0, j].plot(prediction[0, :, j], label='Predicted', linewidth=2, color='blue', alpha=0.8)
            axes[0, j].plot(y_sample[:, j], label='True', linewidth=2, color='red', alpha=0.8)
            
            # Mark peaks
            pred_peak_idx = peak_analysis[component]['pred_peak_idx']
            pred_peak_val = peak_analysis[component]['pred_peak_val']
            true_peak_idx = peak_analysis[component]['true_peak_idx']
            true_peak_val = peak_analysis[component]['true_peak_val']
            
            axes[0, j].plot(pred_peak_idx, pred_peak_val, 'bo', markersize=8, label='Pred Peak')
            axes[0, j].plot(true_peak_idx, true_peak_val, 'ro', markersize=8, label='True Peak')
            
            # Add peak difference to legend
            x_diff = peak_analysis[component]['x_diff']
            y_diff = peak_analysis[component]['y_diff']
            peak_info = f'Δx={x_diff:.1f}%, ΔF={y_diff:.1f}%'
            
            axes[0, j].set_title(f'{component} - {peak_info}')
            axes[0, j].set_xlabel('Position along robot')
            axes[0, j].set_ylabel(f'{component} (N)')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)
            
            # Bottom row: Error
            error = prediction[0, :, j] - y_sample[:, j]
            axes[1, j].plot(error, color='red', linewidth=2)
            axes[1, j].set_title(f'{component} Error (MAE: {np.mean(np.abs(error)):.4f})')
            axes[1, j].set_xlabel('Position along robot')
            axes[1, j].set_ylabel('Error (N)')
            axes[1, j].grid(True, alpha=0.3)
            axes[1, j].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def _visualize_specific_sample(self, x_test, y_test, sample_idx, sample_type="Specific"):
        """Visualize a specific sample prediction with peak detection"""
        
        print(f"\nShowing {sample_type.lower()} sample #{sample_idx + 1} from {x_test.shape[0]} test samples")
        
        # Make prediction for this sample
        x_sample = x_test[sample_idx:sample_idx+1]  # Keep batch dimension
        prediction = self.predict(x_sample)
        y_sample = y_test[sample_idx]
        
        # Calculate metrics for this sample
        mse = np.mean((prediction[0] - y_sample)**2)
        mae = np.mean(np.abs(prediction[0] - y_sample))
        
        # Peak analysis for this sample
        peak_analysis = self.calculate_peak_differences(prediction[0], y_sample)
        
        # Print peak analysis for this sample
        print(f"\nPeak Analysis for {sample_type} Sample #{sample_idx + 1}:")
        for comp in ['Fx', 'Fy', 'Fz']:
            x_diff = peak_analysis[comp]['x_diff']
            y_diff = peak_analysis[comp]['y_diff']
            print(f"  {comp}: Position diff = {x_diff:.1f}%, Force diff = {y_diff:.1f}%")
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'{sample_type} Test Sample #{sample_idx + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}', fontsize=14)
        
        for j, component in enumerate(['Fx', 'Fy', 'Fz']):
            # Top row: Predictions vs True with peaks marked
            axes[0, j].plot(prediction[0, :, j], label='Predicted', linewidth=2, color='blue', alpha=0.8)
            axes[0, j].plot(y_sample[:, j], label='True', linewidth=2, color='red', alpha=0.8)
            
            # Mark peaks
            pred_peak_idx = peak_analysis[component]['pred_peak_idx']
            pred_peak_val = peak_analysis[component]['pred_peak_val']
            true_peak_idx = peak_analysis[component]['true_peak_idx']
            true_peak_val = peak_analysis[component]['true_peak_val']
            
            axes[0, j].plot(pred_peak_idx, pred_peak_val, 'bo', markersize=8, label='Pred Peak')
            axes[0, j].plot(true_peak_idx, true_peak_val, 'ro', markersize=8, label='True Peak')
            
            # Add peak difference to legend
            x_diff = peak_analysis[component]['x_diff']
            y_diff = peak_analysis[component]['y_diff']
            peak_info = f'Δx={x_diff:.1f}%, ΔF={y_diff:.1f}%'
            
            axes[0, j].set_title(f'{component} - {peak_info}')
            axes[0, j].set_xlabel('Position along robot')
            axes[0, j].set_ylabel(f'{component} (N)')
            axes[0, j].legend()
            axes[0, j].grid(True, alpha=0.3)
            
            # Bottom row: Error
            error = prediction[0, :, j] - y_sample[:, j]
            axes[1, j].plot(error, color='red', linewidth=2)
            axes[1, j].set_title(f'{component} Error (MAE: {np.mean(np.abs(error)):.4f})')
            axes[1, j].set_xlabel('Position along robot')
            axes[1, j].set_ylabel('Error (N)')
            axes[1, j].grid(True, alpha=0.3)
            axes[1, j].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self):
        """Plot training history if available"""
        if self.train_losses and self.val_losses:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.train_losses, label=f'Training Loss (Final: {self.train_losses[-1]:.6f})', alpha=0.8, linewidth=2)
            ax.plot(self.val_losses, label=f'Validation Loss (Final: {self.val_losses[-1]:.6f})', alpha=0.8, linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title(f'Training History ({len(self.train_losses)} epochs)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            return fig
        else:
            print("No training history available")
            return None

def load_test_data(npz_file_path="12_1_testing_hans33.npz"):
    """Load test data from npz file"""
    from pathlib import Path
    
    # Try multiple possible file locations
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    possible_files = [
        npz_file_path,
        script_dir / npz_file_path,
        script_dir.parent / npz_file_path,
        Path.cwd() / npz_file_path,
        script_dir.parent / "Hans_new_networks" / npz_file_path,
        Path.cwd() / "Hans_new_networks" / npz_file_path,
    ]
    
    actual_file_path = None
    for filepath in possible_files:
        if Path(filepath).exists():
            actual_file_path = str(filepath)
            break
    
    if actual_file_path is None:
        raise FileNotFoundError(f"Could not find {npz_file_path} in any of the searched locations")
    
    data = np.load(actual_file_path)
    # Use test data keys if available, fallback to train keys for compatibility
    x_key = "x_test" if "x_test" in data else "x_train"
    y_key = "y_test" if "y_test" in data else "y_train"
    x_data = data[x_key]  # Shape: (n_samples, 12, 3)
    y_data = data[y_key]  # Shape: (n_samples, 64, 3)
    
    print(f"Loaded test data:")
    print(f"  Input shape: {x_data.shape} (samples, actuator_tokens, features)")
    print(f"  Output shape: {y_data.shape} (samples, position_tokens, force_xyz)")
    
    return x_data, y_data

# def create_synthetic_test_case():
#     """Create a synthetic test case for demonstration"""
#     print("Creating synthetic test case...")
    
#     # Create actuator data: small deformations
#     x_test = np.zeros((1, 12, 3))  # Single sample
    
#     # Set initial lengths (all equal - straight robot)
#     x_test[0, :, 0] = 4.0  # Initial lengths
    
#     # Create some deformation (bend the robot)
#     for seg in range(4):
#         for act in range(3):
#             idx = seg*3 + act
#             # Create bending: actuator 0 gets shorter, actuator 2 gets longer
#             if act == 0:
#                 deformation = -0.2 * (seg + 1) / 4  # Progressive bending
#             elif act == 2:
#                 deformation = 0.2 * (seg + 1) / 4
#             else:
#                 deformation = 0.0
            
#             x_test[0, idx, 1] = x_test[0, idx, 0] + deformation  # Deformed length
#             x_test[0, idx, 2] = deformation  # Difference
    
#     print("Synthetic test case created:")
#     for seg in range(4):
#         print(f"  Segment {seg+1}:")
#         for act in range(3):
#             idx = seg*3 + act
#             initial, deformed, diff = x_test[0, idx, :]
#             print(f"    Act {act+1}: {initial:.3f} → {deformed:.3f} (Δ={diff:.3f})")
    
#     return x_test

if __name__ == "__main__":
    # Try to find the best available model file
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    possible_models = [
        'complete_transformer_model_24x3_BEST.pth']
    
    model_file = None
    for model_name in possible_models:
        full_path = script_dir / model_name
        if full_path.exists():
            model_file = str(full_path)  # Use full path, not just filename
            print(f"Found model file: {model_name}")
            print(f"Full path: {model_file}")
            break
    
    if model_file is None:
        print("No trained model found! Please train a model first using:")
        print("  python transfomer_model_24.py")
        print("\nLooking for these files:")
        for model_name in possible_models:
            print(f"  - {script_dir / model_name}")
        sys.exit(1)
    
    # Create model loader
    loader = TransformerModelLoader(model_file)
    
    # Load the trained model
    if not loader.load_model():
        print("Failed to load model. Make sure you've trained and saved a model first.")
        sys.exit(1)
    
    # Load test data
    try:
        x_test, y_test = load_test_data("12_1_testing_hans44.npz")
        print(f"\nLoaded {x_test.shape[0]} test samples")
        
        # Calculate overall test metrics on all samples
        print(f"Calculating metrics on all {x_test.shape[0]} samples...")
        predictions = loader.predict(x_test)
        
        # Calculate per-sample MSE and MAE for standard deviation
        sample_mses = np.mean((predictions - y_test)**2, axis=(1, 2))  # MSE per sample
        sample_maes = np.mean(np.abs(predictions - y_test), axis=(1, 2))  # MAE per sample
        
        overall_mse = np.mean(sample_mses)
        overall_mae = np.mean(sample_maes)
        overall_mse_std = np.std(sample_mses)
        overall_mae_std = np.std(sample_maes)
        
        print(f"Overall Test MSE: {overall_mse:.6f} ± {overall_mse_std:.6f}")
        print(f"Overall Test MAE: {overall_mae:.6f} ± {overall_mae_std:.6f}")
        
        # Calculate overall peak analysis
        print("\nCalculating peak analysis...")
        overall_peak_diffs = []
        for i in range(len(predictions)):
            peak_analysis = loader.calculate_peak_differences(predictions[i], y_test[i])
            overall_peak_diffs.append(peak_analysis)
        
        # Calculate and display average peak differences
        print("\nOverall Peak Analysis:")
        for comp in ['Fx', 'Fy', 'Fz']:
            x_diffs = [pd[comp]['x_diff'] for pd in overall_peak_diffs]
            y_diffs = [pd[comp]['y_diff'] for pd in overall_peak_diffs]
            avg_x_diff = np.mean(x_diffs)
            avg_y_diff = np.mean(y_diffs)
            std_x_diff = np.std(x_diffs)
            std_y_diff = np.std(y_diffs)
            print(f"  {comp} - Avg Position Diff: {avg_x_diff:.1f}%±{std_x_diff:.1f}%, Avg Force Diff: {avg_y_diff:.1f}%±{std_y_diff:.1f}%")
        
        # Find the samples with best and worst MSE
        print("\nFinding best and worst samples...")
        sample_mses = []
        for i in range(len(predictions)):
            sample_mse = np.mean((predictions[i] - y_test[i])**2)
            sample_mses.append(sample_mse)
        
        best_sample_idx = np.argmin(sample_mses)
        worst_sample_idx = np.argmax(sample_mses)
        best_sample_mse = sample_mses[best_sample_idx]
        worst_sample_mse = sample_mses[worst_sample_idx]
        
        print(f"Best sample: #{best_sample_idx + 1} with MSE: {best_sample_mse:.6f}")
        print(f"Worst sample: #{worst_sample_idx + 1} with MSE: {worst_sample_mse:.6f}")
        
        # Create the four plots
        print("\nGenerating plots...")
        
        # Plot 1: Training history
        fig1 = loader.plot_training_history()
        
        # Plot 2: Random sample
        fig2 = loader._visualize_random_sample(x_test, y_test)
        
        # Plot 3: Best sample
        fig3 = loader._visualize_specific_sample(x_test, y_test, best_sample_idx, "Best")
        
        # Plot 4: Worst sample
        fig4 = loader._visualize_specific_sample(x_test, y_test, worst_sample_idx, "Worst")
        
        # Show both plots simultaneously
        if fig1 is not None:
            plt.show()
        
    except FileNotFoundError:
        print("\nTest data (12_1_testing_hans33.npz) not found!")
        print("Please make sure the file exists in the Hans_new_networks directory.")
    
    print("\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)
    print("Run again to see a new random sample!")