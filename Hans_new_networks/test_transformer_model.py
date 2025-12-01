import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import the model classes from transformer_model.py
from transformer_model import DeformationToForceTransformer, PositionalEncoding

class TransformerModelLoader:
    """Class for loading and testing trained transformer models"""
    
    def __init__(self, model_path='complete_transformer_model.pth'):
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
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Extract configuration
            self.model_config = model_data['model_config']
            self.train_losses = model_data.get('train_losses', [])
            self.val_losses = model_data.get('val_losses', [])
            
            # Create model with saved configuration
            self.model = DeformationToForceTransformer(
                input_dim=self.model_config['input_dim'],
                output_dim=self.model_config['output_dim'],
                output_seq_len=self.model_config.get('output_seq_len', 192),  # Default for backward compatibility
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
            predictions: numpy array of shape (n_samples, 192, 3) or (192, 3) for single sample
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
                'mae_fz': mae_components[2]
            })
            
            print(f"\nEvaluation Results:")
            print(f"  Overall MSE: {mse:.6f}")
            print(f"  Overall MAE: {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")
            print(f"  Component MSE - Fx: {mse_components[0]:.6f}, Fy: {mse_components[1]:.6f}, Fz: {mse_components[2]:.6f}")
            print(f"  Component MAE - Fx: {mae_components[0]:.6f}, Fy: {mae_components[1]:.6f}, Fz: {mae_components[2]:.6f}")
        
        return results
    
    def _visualize_random_sample(self, x_test, y_test):
        """Visualize a single random sample prediction"""
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
        
        # Create the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'Random Test Sample #{sample_idx + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}', fontsize=14)
        
        for j, component in enumerate(['Fx', 'Fy', 'Fz']):
            # Top row: Predictions vs True
            axes[0, j].plot(prediction[0, :, j], label='Predicted', linewidth=2, color='blue', alpha=0.8)
            axes[0, j].plot(y_sample[:, j], label='True', linewidth=2, color='red', alpha=0.8)
            axes[0, j].set_title(f'{component}')
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

def load_test_data(npz_file_path="11_31_testing_hans.npz"):
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

def create_synthetic_test_case():
    """Create a synthetic test case for demonstration"""
    print("Creating synthetic test case...")
    
    # Create actuator data: small deformations
    x_test = np.zeros((1, 12, 3))  # Single sample
    
    # Set initial lengths (all equal - straight robot)
    x_test[0, :, 0] = 4.0  # Initial lengths
    
    # Create some deformation (bend the robot)
    for seg in range(4):
        for act in range(3):
            idx = seg*3 + act
            # Create bending: actuator 0 gets shorter, actuator 2 gets longer
            if act == 0:
                deformation = -0.2 * (seg + 1) / 4  # Progressive bending
            elif act == 2:
                deformation = 0.2 * (seg + 1) / 4
            else:
                deformation = 0.0
            
            x_test[0, idx, 1] = x_test[0, idx, 0] + deformation  # Deformed length
            x_test[0, idx, 2] = deformation  # Difference
    
    print("Synthetic test case created:")
    for seg in range(4):
        print(f"  Segment {seg+1}:")
        for act in range(3):
            idx = seg*3 + act
            initial, deformed, diff = x_test[0, idx, :]
            print(f"    Act {act+1}: {initial:.3f} → {deformed:.3f} (Δ={diff:.3f})")
    
    return x_test

if __name__ == "__main__":
    # Create model loader
    loader = TransformerModelLoader('complete_transformer_model.pth')
    
    # Load the trained model
    if not loader.load_model():
        print("Failed to load model. Make sure you've trained and saved a model first.")
        sys.exit(1)
    
    # Load test data
    try:
        x_test, y_test = load_test_data("11_31_testing_hans.npz")
        print(f"\nLoaded {x_test.shape[0]} test samples")
        
        # Calculate overall test metrics (on subset for speed)
        test_subset_size = min(500, x_test.shape[0])
        x_subset = x_test[:test_subset_size]
        y_subset = y_test[:test_subset_size]
        
        print(f"Calculating metrics on {test_subset_size} samples...")
        predictions = loader.predict(x_subset)
        overall_mse = np.mean((predictions - y_subset)**2)
        overall_mae = np.mean(np.abs(predictions - y_subset))
        
        print(f"Overall Test MSE: {overall_mse:.6f}")
        print(f"Overall Test MAE: {overall_mae:.6f}")
        
        # Create the two plots
        print("\nGenerating plots...")
        
        # Plot 1: Training history
        fig1 = loader.plot_training_history()
        
        # Plot 2: Random sample
        fig2 = loader._visualize_random_sample(x_test, y_test)
        
        # Show both plots simultaneously
        if fig1 is not None:
            plt.show()
        
    except FileNotFoundError:
        print("\nTest data (11_30_testing.npz) not found!")
        print("Please make sure the file exists in the Hans_new_networks directory.")
    
    print("\n" + "="*50)
    print("TESTING COMPLETE")
    print("="*50)
    print("Run again to see a new random sample!")