"""
Neural Network Model Evaluation Script

This script loads a pre-trained neural network model and evaluates it on test data.
It can load models saved by the Feed_forward_nn_training.py script and provides
detailed performance analysis including predictions, error statistics, and visualizations.

Usage:
    python evaluate_trained_model.py

The script will automatically look for saved model files and prompt you to select one.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time


class ModelEvaluator:
    """
    Class for loading and evaluating trained neural network models.
    """
    
    def __init__(self):
        self.model_data = None
        self.normalization_params = None
        self.network_config = None
        self.s_force = None
        
    def load_model(self, model_path):
        """
        Load a trained model from file.
        
        Parameters:
        model_path (str): Path to the saved model file (.npz)
        
        Returns:
        bool: True if loading successful, False otherwise
        """
        try:
            print(f"Loading model from: {model_path}")
            self.model_data = np.load(model_path, allow_pickle=True)
            
            # Extract model components - handle variable architecture
            self.w1 = self.model_data['w1']
            self.b1 = self.model_data['b1']
            
            # Load additional weights based on architecture
            self.w2 = self.model_data['w2']
            self.b2 = self.model_data['b2']
            
            # Check if third layer exists
            if 'w3' in self.model_data:
                self.w3 = self.model_data['w3']
                self.b3 = self.model_data['b3']
            else:
                self.w3 = None
                self.b3 = None
                
            # Check if fourth layer exists (for 3-hidden-layer networks)
            if 'w4' in self.model_data:
                self.w4 = self.model_data['w4']
                self.b4 = self.model_data['b4']
            else:
                self.w4 = None
                self.b4 = None
            
            # Extract normalization parameters
            self.x_mean = self.model_data['x_mean']
            self.x_std = self.model_data['x_std']
            self.y_mean = self.model_data['y_mean']
            self.y_std = self.model_data['y_std']
            
            # Extract network configuration
            self.network_config = self.model_data['network_config'].item()
            
            # Determine architecture
            if 'num_hidden_layers' in self.network_config:
                self.num_hidden_layers = self.network_config['num_hidden_layers']
            elif 'use_two_layers' in self.network_config:
                self.num_hidden_layers = 2 if self.network_config['use_two_layers'] else 1
            else:
                # Legacy support - assume single layer if not specified
                self.num_hidden_layers = 1
            
            # Extract other data
            self.s_force = self.model_data['s_force']
            
            # Training history (if available)
            if 'train_losses' in self.model_data:
                self.train_losses = self.model_data['train_losses']
            if 'test_losses' in self.model_data:
                self.test_losses = self.model_data['test_losses']
            
            print("Model loaded successfully!")
            print(f"  Input size: {self.network_config['input_size']}")
            
            # Display architecture based on number of layers
            if self.num_hidden_layers == 1:
                hidden_size = self.network_config.get('hidden_size', self.network_config.get('hidden_size1', 'Unknown'))
                print(f"  Architecture: {self.network_config['input_size']} → {hidden_size} → {self.network_config['output_size']}")
            elif self.num_hidden_layers == 2:
                h1 = self.network_config.get('hidden_size1', 'Unknown')
                h2 = self.network_config.get('hidden_size2', 'Unknown')
                print(f"  Architecture: {self.network_config['input_size']} → {h1} → {h2} → {self.network_config['output_size']}")
            elif self.num_hidden_layers == 3:
                h1 = self.network_config.get('hidden_size1', 'Unknown')
                h2 = self.network_config.get('hidden_size2', 'Unknown')
                h3 = self.network_config.get('hidden_size3', 'Unknown')
                print(f"  Architecture: {self.network_config['input_size']} → {h1} → {h2} → {h3} → {self.network_config['output_size']}")
            
            print(f"  Learning rate: {self.network_config['learning_rate']}")
            print(f"  Hidden layers: {self.num_hidden_layers}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def tanh_activation(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def linear_activation(self, x):
        """Linear activation function"""
        return x
    
    def forward_pass(self, x):
        """
        Perform forward pass through the loaded network.
        
        Parameters:
        x (ndarray): Input data of shape (input_size, batch_size)
        
        Returns:
        ndarray: Network predictions
        """
        # First hidden layer (always present)
        z1 = self.w1 @ x + self.b1
        a1 = self.tanh_activation(z1)
        
        if self.num_hidden_layers == 1:
            # Single layer: input → h1 → output
            z2 = self.w2 @ a1 + self.b2
            a2 = self.linear_activation(z2)
            return a2
            
        elif self.num_hidden_layers == 2:
            # Two layers: input → h1 → h2 → output
            z2 = self.w2 @ a1 + self.b2
            a2 = self.tanh_activation(z2)
            
            z3 = self.w3 @ a2 + self.b3
            a3 = self.linear_activation(z3)
            return a3
            
        elif self.num_hidden_layers == 3:
            # Three layers: input → h1 → h2 → h3 → output
            z2 = self.w2 @ a1 + self.b2
            a2 = self.tanh_activation(z2)
            
            z3 = self.w3 @ a2 + self.b3
            a3 = self.tanh_activation(z3)
            
            z4 = self.w4 @ a3 + self.b4
            a4 = self.linear_activation(z4)
            return a4
    
    def predict(self, x):
        """
        Make predictions using the loaded model.
        
        Parameters:
        x (ndarray): Input data (normalized)
        
        Returns:
        ndarray: Predictions
        """
        return self.forward_pass(x)
    
    def normalize_inputs(self, x):
        """Normalize inputs using saved parameters"""
        x_normalized = (x - self.x_mean) / self.x_std
        return np.clip(x_normalized, -5, 5)
    
    def denormalize_outputs(self, y):
        """Denormalize outputs using saved parameters"""
        return y * self.y_std + self.y_mean
    
    def load_test_data(self, npz_file_path="simulation_results_all.npz", train_split=0.5):
        """
        Load and prepare test data (same as training script).
        
        Parameters:
        npz_file_path (str): Path to the simulation data
        train_split (float): Train/test split ratio (should match training)
        
        Returns:
        tuple: (x_test_raw, y_test_raw, x_test_norm, y_test_norm)
        """
        print(f"Loading test data from {npz_file_path}...")
        
        # Try multiple possible file locations
        script_dir = Path(__file__).parent
        possible_files = [
            npz_file_path,
            script_dir / npz_file_path,
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
        
        # Load the data
        npz_data = np.load(actual_file_path, allow_pickle=True)
        simulation_results = npz_data['data']
        
        print(f"Total simulations: {len(simulation_results)}")
        
        # Extract features and targets (same as training)
        input_features = []
        output_forces = []
        
        for sim in simulation_results:
            # All actuator data for segments 1-4 (24 features total)
            inputs = [
                sim['L1_input_1'], sim['L1_def_1'], sim['L2_input_1'], sim['L2_def_1'], sim['L3_input_1'], sim['L3_def_1'],
                sim['L1_input_2'], sim['L1_def_2'], sim['L2_input_2'], sim['L2_def_2'], sim['L3_input_2'], sim['L3_def_2'],
                sim['L1_input_3'], sim['L1_def_3'], sim['L2_input_3'], sim['L2_def_3'], sim['L3_input_3'], sim['L3_def_3'],
                sim['L1_input_4'], sim['L1_def_4'], sim['L2_input_4'], sim['L2_def_4'], sim['L3_input_4'], sim['L3_def_4']
            ]
            
            # Concatenated force distributions
            forces = np.concatenate([
                sim['f_dist_fx'],  # 200 values - Force X
                sim['f_dist_fy'],  # 200 values - Force Y
                sim['f_dist_fz']   # 200 values - Force Z
            ])
            
            input_features.append(inputs)
            output_forces.append(forces)
        
        X = np.array(input_features)
        Y = np.array(output_forces)
        
        # Filter extreme values (same as training)
        valid_mask = (np.abs(Y).max(axis=1) < 1.0) & (np.abs(X).max(axis=1) < 1000)
        X = X[valid_mask]
        Y = Y[valid_mask]
        
        # Use same random seed as training for consistent split
        indices = np.arange(len(X))
        np.random.seed(42)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        
        # Extract test set (same split as training)
        split_idx = int(len(X) * train_split)
        x_test_raw = X[split_idx:]  # Raw test inputs
        y_test_raw = Y[split_idx:]  # Raw test outputs
        
        # Normalize using saved parameters
        x_test_norm = self.normalize_inputs(x_test_raw.T)  # Shape: (24, n_test)
        y_test_norm = (y_test_raw.T - self.y_mean) / self.y_std  # Shape: (600, n_test)
        y_test_norm = np.clip(y_test_norm, -5, 5)
        
        npz_data.close()
        
        print(f"Test set: {x_test_norm.shape[1]} samples")
        print(f"Input shape: {x_test_norm.shape}")
        print(f"Output shape: {y_test_norm.shape}")
        
        return x_test_raw, y_test_raw, x_test_norm, y_test_norm
    
    def evaluate_model(self, x_test_norm, y_test_norm, x_test_raw=None, y_test_raw=None):
        """
        Evaluate the model on test data and return detailed metrics.
        
        Parameters:
        x_test_norm: Normalized test inputs
        y_test_norm: Normalized test outputs
        x_test_raw: Raw test inputs (for reference)
        y_test_raw: Raw test outputs (for denormalized comparisons)
        
        Returns:
        dict: Evaluation metrics and predictions
        """
        print("Evaluating model on test set...")
        
        # Make predictions
        predictions_norm = self.predict(x_test_norm)
        
        # Calculate normalized loss
        error_norm = predictions_norm - y_test_norm
        mse_loss_norm = np.mean(error_norm ** 2)
        mae_loss_norm = np.mean(np.abs(error_norm))
        
        # Denormalize for real-world metrics
        predictions_real = self.denormalize_outputs(predictions_norm)
        
        if y_test_raw is not None:
            error_real = predictions_real.T - y_test_raw
            mse_loss_real = np.mean(error_real ** 2)
            mae_loss_real = np.mean(np.abs(error_real))
            rmse_loss_real = np.sqrt(mse_loss_real)
        else:
            # If no raw data provided, denormalize the normalized targets
            y_test_real = self.denormalize_outputs(y_test_norm)
            error_real = predictions_real - y_test_real
            mse_loss_real = np.mean(error_real ** 2)
            mae_loss_real = np.mean(np.abs(error_real))
            rmse_loss_real = np.sqrt(mse_loss_real)
        
        # Component-wise analysis (Fx, Fy, Fz)
        fx_error = error_real[:, :200] if len(error_real.shape) > 1 else error_real[:200]
        fy_error = error_real[:, 200:400] if len(error_real.shape) > 1 else error_real[200:400]
        fz_error = error_real[:, 400:600] if len(error_real.shape) > 1 else error_real[400:600]
        
        fx_rmse = np.sqrt(np.mean(fx_error ** 2))
        fy_rmse = np.sqrt(np.mean(fy_error ** 2))
        fz_rmse = np.sqrt(np.mean(fz_error ** 2))
        
        results = {
            'mse_loss_norm': mse_loss_norm,
            'mae_loss_norm': mae_loss_norm,
            'mse_loss_real': mse_loss_real,
            'mae_loss_real': mae_loss_real,
            'rmse_loss_real': rmse_loss_real,
            'fx_rmse': fx_rmse,
            'fy_rmse': fy_rmse,
            'fz_rmse': fz_rmse,
            'predictions_norm': predictions_norm,
            'predictions_real': predictions_real,
            'error_real': error_real,
            'n_test_samples': x_test_norm.shape[1]
        }
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Test samples: {results['n_test_samples']}")
        print(f"Normalized MSE Loss: {mse_loss_norm:.6f}")
        print(f"Real-world RMSE: {rmse_loss_real:.6f} N")
        print(f"Real-world MAE:  {mae_loss_real:.6f} N")
        print("\nComponent-wise RMSE:")
        print(f"  Fx (Force X): {fx_rmse:.6f} N")
        print(f"  Fy (Force Y): {fy_rmse:.6f} N") 
        print(f"  Fz (Force Z): {fz_rmse:.6f} N")
        
        # Performance assessment
        if mse_loss_norm < 0.05:
            performance = "EXCELLENT"
        elif mse_loss_norm < 0.2:
            performance = "VERY GOOD"
        elif mse_loss_norm < 0.5:
            performance = "GOOD"
        elif mse_loss_norm < 0.8:
            performance = "ACCEPTABLE"
        else:
            performance = "POOR"
        
        print(f"\nPerformance Assessment: {performance}")
        print("="*60)
        
        return results
    
    def plot_evaluation_results(self, results, sample_indices=None, save_plots=True):
        """
        Create comprehensive evaluation plots.
        
        Parameters:
        results (dict): Results from evaluate_model
        sample_indices (list): Specific samples to plot (default: random selection)
        save_plots (bool): Whether to save plots to files
        """
        predictions_real = results['predictions_real']
        error_real = results['error_real']
        
        if sample_indices is None:
            # Select a few random samples for detailed analysis
            n_samples = min(5, predictions_real.shape[1] if len(predictions_real.shape) > 1 else 1)
            sample_indices = np.random.choice(results['n_test_samples'], n_samples, replace=False)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Training history (if available)
        if hasattr(self, 'train_losses') and hasattr(self, 'test_losses'):
            plt.subplot(3, 3, 1)
            plt.plot(self.train_losses, 'b-', label='Training Loss', alpha=0.7)
            if len(self.test_losses) > 0:
                test_epochs = list(range(0, len(self.test_losses) * 10, 10))
                plt.plot(test_epochs, self.test_losses, 'r-', label='Test Loss', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        # Plot 2-4: Sample predictions for each force component
        colors = ['red', 'green', 'blue']
        components = ['Fx', 'Fy', 'Fz']
        
        for i, (comp, color) in enumerate(zip(components, colors)):
            plt.subplot(3, 3, 2 + i)
            
            sample_idx = sample_indices[0]
            if len(predictions_real.shape) > 1:
                pred = predictions_real[:, sample_idx]
                actual = (results['predictions_real'][:, sample_idx] - error_real[sample_idx, :])
            else:
                pred = predictions_real
                actual = pred - error_real
            
            start_idx = i * 200
            end_idx = (i + 1) * 200
            
            plt.plot(self.s_force, actual[start_idx:end_idx], '-', color=color, 
                    label=f'Actual {comp}', linewidth=2, alpha=0.8)
            plt.plot(self.s_force, pred[start_idx:end_idx], '--', color='black',
                    label=f'Predicted {comp}', linewidth=2, alpha=0.8)
            
            plt.xlabel('Arc Length s')
            plt.ylabel('Force (N)')
            plt.title(f'{comp} Component (Sample {sample_idx})')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Error distribution
        plt.subplot(3, 3, 5)
        if len(error_real.shape) > 1:
            all_errors = error_real.flatten()
        else:
            all_errors = error_real
        
        plt.hist(all_errors, bins=50, alpha=0.7, edgecolor='black', density=True)
        plt.xlabel('Prediction Error (N)')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)
        plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}')
        plt.axvline(mean_error + std_error, color='orange', linestyle=':', label=f'+1σ: {mean_error + std_error:.4f}')
        plt.axvline(mean_error - std_error, color='orange', linestyle=':', label=f'-1σ: {mean_error - std_error:.4f}')
        plt.legend()
        
        # Plot 6: Component RMSE comparison
        plt.subplot(3, 3, 6)
        rmse_values = [results['fx_rmse'], results['fy_rmse'], results['fz_rmse']]
        bars = plt.bar(components, rmse_values, color=colors, alpha=0.7)
        plt.ylabel('RMSE (N)')
        plt.title('Component-wise RMSE')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom')
        
        # Plot 7-9: Prediction vs Actual scatter plots for each component
        for i, (comp, color) in enumerate(zip(components, colors)):
            plt.subplot(3, 3, 7 + i)
            
            start_idx = i * 200
            end_idx = (i + 1) * 200
            
            if len(predictions_real.shape) > 1:
                pred_comp = predictions_real[start_idx:end_idx, :].flatten()
                actual_comp = (predictions_real[start_idx:end_idx, :] - 
                              error_real[:, start_idx:end_idx].T).flatten()
            else:
                pred_comp = predictions_real[start_idx:end_idx]
                actual_comp = pred_comp - error_real[start_idx:end_idx]
            
            plt.scatter(actual_comp, pred_comp, alpha=0.5, color=color, s=1)
            
            # Perfect prediction line
            min_val = min(actual_comp.min(), pred_comp.min())
            max_val = max(actual_comp.max(), pred_comp.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
            
            plt.xlabel(f'Actual {comp} (N)')
            plt.ylabel(f'Predicted {comp} (N)')
            plt.title(f'{comp}: Predicted vs Actual')
            plt.grid(True, alpha=0.3)
            
            # Calculate R²
            correlation_matrix = np.corrcoef(actual_comp, pred_comp)
            r_squared = correlation_matrix[0, 1] ** 2
            plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved to 'model_evaluation_results.png'")
        
        plt.show()
        
        return fig


def find_model_files():
    """Find available model files in the current directory and subdirectories."""
    model_files = []
    
    # Search in current directory and Simulations subdirectory
    search_dirs = [Path.cwd(), Path.cwd() / "Simulations"]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            # Look for .npz files that might be models
            for file_path in search_dir.glob("*.npz"):
                if file_path.name != "simulation_results_all.npz":  # Exclude data file
                    model_files.append(file_path)
    
    return model_files


def main():
    """Main evaluation function."""
    print("Neural Network Model Evaluation")
    print("="*50)
    
    # Find available model files
    model_files = find_model_files()
    
    if not model_files:
        print("No model files found!")
        print("Expected files: trained_model_single.npz, trained_model_best.npz")
        print("Please run the training script first to generate model files.")
        return
    
    # Display available models
    print("Available model files:")
    for i, model_file in enumerate(model_files):
        print(f"  {i+1}. {model_file.name} ({model_file.parent.name})")
    
    # Select model
    if len(model_files) == 1:
        selected_model = model_files[0]
        print(f"\nUsing: {selected_model.name}")
    else:
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(model_files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    selected_model = model_files[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(model_files)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Load and evaluate model
    evaluator = ModelEvaluator()
    
    if not evaluator.load_model(str(selected_model)):
        print("Failed to load model!")
        return
    
    try:
        # Load test data
        x_test_raw, y_test_raw, x_test_norm, y_test_norm = evaluator.load_test_data()
        
        # Evaluate model
        results = evaluator.evaluate_model(x_test_norm, y_test_norm, x_test_raw, y_test_raw)
        
        # Create plots - using specific sample indices for consistent evaluation
        sample_indices = [156, 287, 394]  # Different samples to evaluate
        evaluator.plot_evaluation_results(results, sample_indices=sample_indices)
        
        print("\nEvaluation complete!")
        print("Check 'model_evaluation_results.png' for detailed plots.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()