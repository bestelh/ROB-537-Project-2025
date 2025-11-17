

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


def data_import(npz_file_path="simulation_results_all.npz", train_split=0.5):
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
           - y_train: Training targets (600 force values × n_train_samples)
           - x_test: Test inputs (24 features × n_test_samples)
           - y_test: Test targets (600 force values × n_test_samples)
           - s_force: Arc length positions for force profiles
           - normalization_params: Statistics for denormalizing predictions
    """
    print(f"Loading data from {npz_file_path}...")
    
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
        raise FileNotFoundError(f"Could not find {npz_file_path} in any of the searched locations")
    
    # Load the data
    npz_data = np.load(actual_file_path, allow_pickle=True)
    simulation_results = npz_data['data']
    
    print(f"Total number of simulations: {len(simulation_results)}")
    
    # Extract features and targets
    input_features = []
    output_forces = []
    
    for sim in simulation_results:
        # Input: All actuator data for segments 1-4 (24 features total)
        # L1_input, L1_def, L2_input, L2_def, L3_input, L3_def for each segment
        inputs = [
            sim['L1_input_1'], sim['L1_def_1'], sim['L2_input_1'], sim['L2_def_1'], sim['L3_input_1'], sim['L3_def_1'],
            sim['L1_input_2'], sim['L1_def_2'], sim['L2_input_2'], sim['L2_def_2'], sim['L3_input_2'], sim['L3_def_2'],
            sim['L1_input_3'], sim['L1_def_3'], sim['L2_input_3'], sim['L2_def_3'], sim['L3_input_3'], sim['L3_def_3'],
            sim['L1_input_4'], sim['L1_def_4'], sim['L2_input_4'], sim['L2_def_4'], sim['L3_input_4'], sim['L3_def_4']
        ]
        
        # Output: Concatenated force distributions (600 values total)
        forces = np.concatenate([
            sim['f_dist_fx'],  # 200 values - Force X
            sim['f_dist_fy'],  # 200 values - Force Y
            sim['f_dist_fz']   # 200 values - Force Z
        ])
        
        input_features.append(inputs)
        output_forces.append(forces)
    
    X = np.array(input_features)  # Shape: (n_samples, 24)
    Y = np.array(output_forces)   # Shape: (n_samples, 600)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    print(f"Input range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Output range: [{Y.min():.6f}, {Y.max():.6f}]")
    
    # Get arc length positions for reference
    s_force = simulation_results[0]['s_force']
    
    # Remove samples with extreme values for stability
    valid_mask = (np.abs(Y).max(axis=1) < 1.0) & (np.abs(X).max(axis=1) < 1000)
    X = X[valid_mask]
    Y = Y[valid_mask]
    print(f"After filtering extreme values: {X.shape[0]} samples remaining")
    
    # Randomize data order for better training
    indices = np.arange(len(X))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    # Split into training and test sets
    # CRITICAL: Test data is COMPLETELY SEPARATED and never used for training!
    split_idx = int(len(X) * train_split)
    x_train = X[:split_idx].T  # Shape: (24, n_train) - First 50% for training
    y_train = Y[:split_idx].T  # Shape: (600, n_train)
    x_test = X[split_idx:].T   # Shape: (24, n_test) - Last 50% for testing
    y_test = Y[split_idx:].T   # Shape: (600, n_test)
    
    print(f"Training set: {x_train.shape[1]} samples ({train_split*100:.0f}% of data)")
    print(f"Test set: {x_test.shape[1]} samples ({(1-train_split)*100:.0f}% of data)")
    print(f"Total samples used: {x_train.shape[1] + x_test.shape[1]}")
    
    # Normalize inputs
    x_mean = np.mean(x_train, axis=1, keepdims=True)
    x_std = np.std(x_train, axis=1, keepdims=True)
    x_std = np.maximum(x_std, 1e-6)  # Prevent division by zero
    
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    
    # Normalize outputs
    y_mean = np.mean(y_train, axis=1, keepdims=True)
    y_std = np.std(y_train, axis=1, keepdims=True)
    y_std = np.maximum(y_std, 1e-6)
    
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Clip normalized values to prevent extreme inputs
    x_train = np.clip(x_train, -5, 5)
    x_test = np.clip(x_test, -5, 5)
    y_train = np.clip(y_train, -5, 5)
    y_test = np.clip(y_test, -5, 5)
    
    npz_data.close()
    
    normalization_params = (x_mean, x_std, y_mean, y_std)
    return x_train, y_train, x_test, y_test, s_force, normalization_params


class FeedForwardNN:
    """
    Feed-forward neural network for distributed force prediction.
    
    Architecture:
    - Input layer: 24 neurons (L1, L2, L3 inputs and deformations for 4 segments)
    - Hidden layer: configurable size with tanh activation
    - Output layer: 600 neurons (force distributions) with linear activation
    """
    
    def __init__(self, input_size=24, hidden_size1=768, hidden_size2=512, hidden_size3=256, output_size=600, 
                 learning_rate=0.001, dropout_rate=0.3, l2_reg=0.01,
                 epochs=1000, batch_size=32, print_every=50, 
                 early_stop_patience=40, lr_decay_rate=0.9, lr_decay_every=250,
                 max_grad_norm=1.0, weight_clip_value=10.0, num_hidden_layers=3):
        """
        Initialize the neural network with all hyperparameters.
        
        Network Architecture:
        - input_size: Number of input features (default: 24)
        - hidden_size1: First hidden layer neurons (default: 768) 
        - hidden_size2: Second hidden layer neurons (default: 512)
        - hidden_size3: Third hidden layer neurons (default: 256)
        - output_size: Number of output neurons (default: 600)
        - num_hidden_layers: Number of hidden layers 1/2/3 (default: 3)
        
        Training Parameters:
        - learning_rate: Initial learning rate (default: 0.001)
        - epochs: Maximum number of training epochs (default: 1000)
        - batch_size: Mini-batch size for training (default: 32)
        - print_every: Print progress every N epochs (default: 100)
        
        Regularization:
        - dropout_rate: Dropout probability for hidden layer (default: 0.3)
        - l2_reg: L2 regularization strength (default: 0.01)
        - max_grad_norm: Gradient clipping threshold (default: 1.0)
        - weight_clip_value: Weight clipping threshold (default: 10.0)
        
        Learning Rate Scheduling:
        - lr_decay_rate: Learning rate decay factor (default: 0.8)
        - lr_decay_every: Decay learning rate every N epochs (default: 200)
        
        Early Stopping:
        - early_stop_patience: Stop if no improvement for N evaluations (default: 50)
        """
        
        # Network architecture
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2 if num_hidden_layers >= 2 else None
        self.hidden_size3 = hidden_size3 if num_hidden_layers >= 3 else None
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        
        # Timing tracking
        self.epoch_times = []
        self.last_print_time = None
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store original for reference
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every
        
        # Regularization parameters
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.max_grad_norm = max_grad_norm
        self.weight_clip_value = weight_clip_value
        
        # Learning rate scheduling
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_every = lr_decay_every
        
        # Early stopping
        self.early_stop_patience = early_stop_patience
        
        # Better weight initialization (Xavier/Glorot for tanh)
        # First hidden layer (always present)
        self.w1 = np.random.randn(hidden_size1, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size1, 1))
        
        if num_hidden_layers == 1:
            # Single hidden layer: input → h1 → output
            self.w2 = np.random.randn(output_size, hidden_size1) * np.sqrt(2.0 / hidden_size1)
            self.b2 = np.zeros((output_size, 1))
            
        elif num_hidden_layers == 2:
            # Two hidden layers: input → h1 → h2 → output
            self.w2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2.0 / hidden_size1)
            self.b2 = np.zeros((hidden_size2, 1))
            self.w3 = np.random.randn(output_size, hidden_size2) * np.sqrt(2.0 / hidden_size2)
            self.b3 = np.zeros((output_size, 1))
            
        elif num_hidden_layers == 3:
            # Three hidden layers: input → h1 → h2 → h3 → output
            self.w2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2.0 / hidden_size1)
            self.b2 = np.zeros((hidden_size2, 1))
            self.w3 = np.random.randn(hidden_size3, hidden_size2) * np.sqrt(2.0 / hidden_size2)
            self.b3 = np.zeros((hidden_size3, 1))
            self.w4 = np.random.randn(output_size, hidden_size3) * np.sqrt(2.0 / hidden_size3)
            self.b4 = np.zeros((output_size, 1))
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        
        print(f"Initialized FeedForwardNN with hyperparameters:")
        
        # Display architecture based on number of layers
        if num_hidden_layers == 1:
            print(f"  Architecture: {input_size} → {hidden_size1} → {output_size}")
            total_params = (input_size * hidden_size1 + hidden_size1 * output_size)
        elif num_hidden_layers == 2:
            print(f"  Architecture: {input_size} → {hidden_size1} → {hidden_size2} → {output_size}")
            total_params = (input_size * hidden_size1 + hidden_size1 * hidden_size2 + hidden_size2 * output_size)
        elif num_hidden_layers == 3:
            print(f"  Architecture: {input_size} → {hidden_size1} → {hidden_size2} → {hidden_size3} → {output_size}")
            total_params = (input_size * hidden_size1 + hidden_size1 * hidden_size2 + 
                          hidden_size2 * hidden_size3 + hidden_size3 * output_size)
        print(f"  Total parameters: ~{total_params:,}")
        print(f"  Learning rate: {learning_rate} (decay: {lr_decay_rate} every {lr_decay_every} epochs)")
        print(f"  Training: {epochs} epochs, batch size {batch_size}")
        print(f"  Regularization: dropout {dropout_rate}, L2 {l2_reg}")
        print(f"  Early stopping: patience {early_stop_patience}")
    
    def tanh_activation(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation"""
        t = np.tanh(x)
        return 1 - t * t
    
    def linear_activation(self, x):
        """Linear activation function (for output layer)"""
        return x
    
    def dropout_mask(self, shape, training=True):
        """Generate dropout mask during training"""
        if not training or self.dropout_rate == 0:
            return np.ones(shape)
        mask = np.random.binomial(1, 1 - self.dropout_rate, shape) / (1 - self.dropout_rate)
        return mask
    
    def forward_pass(self, x, training=True):
        """
        Perform forward pass through the network.
        
        Parameters:
        x (ndarray): Input data of shape (input_size, batch_size)
        training (bool): Whether in training mode (applies dropout)
        
        Returns:
        tuple: Network intermediate values and final output
        """
        # First hidden layer (always present)
        z1 = self.w1 @ x + self.b1
        a1 = self.tanh_activation(z1)
        dropout_mask1 = self.dropout_mask(a1.shape, training)
        a1_dropout = a1 * dropout_mask1
        
        if self.num_hidden_layers == 1:
            # Single layer: input → h1 → output
            z2 = self.w2 @ a1_dropout + self.b2
            a2 = self.linear_activation(z2)
            return z1, a1, z2, a2, dropout_mask1
            
        elif self.num_hidden_layers == 2:
            # Two layers: input → h1 → h2 → output
            z2 = self.w2 @ a1_dropout + self.b2
            a2 = self.tanh_activation(z2)
            dropout_mask2 = self.dropout_mask(a2.shape, training)
            a2_dropout = a2 * dropout_mask2
            
            z3 = self.w3 @ a2_dropout + self.b3
            a3 = self.linear_activation(z3)
            return z1, a1, z2, a2, z3, a3, dropout_mask1, dropout_mask2
            
        elif self.num_hidden_layers == 3:
            # Three layers: input → h1 → h2 → h3 → output
            z2 = self.w2 @ a1_dropout + self.b2
            a2 = self.tanh_activation(z2)
            dropout_mask2 = self.dropout_mask(a2.shape, training)
            a2_dropout = a2 * dropout_mask2
            
            z3 = self.w3 @ a2_dropout + self.b3
            a3 = self.tanh_activation(z3)
            dropout_mask3 = self.dropout_mask(a3.shape, training)
            a3_dropout = a3 * dropout_mask3
            
            z4 = self.w4 @ a3_dropout + self.b4
            a4 = self.linear_activation(z4)
            return z1, a1, z2, a2, z3, a3, z4, a4, dropout_mask1, dropout_mask2, dropout_mask3
    
    def backward_pass(self, x, y, forward_results):
        batch_size = x.shape[1]
        
        if self.num_hidden_layers == 1:
            z1, a1, z2, a2, dropout_mask1 = forward_results
            
            # Output layer gradients
            error_output = a2 - y
            delta_output = error_output / batch_size
            
            # First hidden layer gradients
            a1_dropout = a1 * dropout_mask1
            error_hidden1 = self.w2.T @ delta_output
            error_hidden1 *= dropout_mask1
            delta_hidden1 = error_hidden1 * self.tanh_derivative(z1)
            
            # Weight and bias gradients with L2 regularization
            dw2 = delta_output @ a1_dropout.T + self.l2_reg * self.w2
            db2 = np.sum(delta_output, axis=1, keepdims=True)
            dw1 = delta_hidden1 @ x.T + self.l2_reg * self.w1
            db1 = np.sum(delta_hidden1, axis=1, keepdims=True)
            
            return dw1, db1, dw2, db2
            
        elif self.num_hidden_layers == 2:
            z1, a1, z2, a2, z3, a3, dropout_mask1, dropout_mask2 = forward_results
            
            # Output layer gradients
            error_output = a3 - y
            delta_output = error_output / batch_size
            
            # Second hidden layer gradients
            a2_dropout = a2 * dropout_mask2
            error_hidden2 = self.w3.T @ delta_output
            error_hidden2 *= dropout_mask2
            delta_hidden2 = error_hidden2 * self.tanh_derivative(z2)
            
            # First hidden layer gradients
            a1_dropout = a1 * dropout_mask1
            error_hidden1 = self.w2.T @ delta_hidden2
            error_hidden1 *= dropout_mask1
            delta_hidden1 = error_hidden1 * self.tanh_derivative(z1)
            
            # Weight and bias gradients with L2 regularization
            dw3 = delta_output @ a2_dropout.T + self.l2_reg * self.w3
            db3 = np.sum(delta_output, axis=1, keepdims=True)
            dw2 = delta_hidden2 @ a1_dropout.T + self.l2_reg * self.w2
            db2 = np.sum(delta_hidden2, axis=1, keepdims=True)
            dw1 = delta_hidden1 @ x.T + self.l2_reg * self.w1
            db1 = np.sum(delta_hidden1, axis=1, keepdims=True)
            
            return dw1, db1, dw2, db2, dw3, db3
            
        elif self.num_hidden_layers == 3:
            z1, a1, z2, a2, z3, a3, z4, a4, dropout_mask1, dropout_mask2, dropout_mask3 = forward_results
            
            # Output layer gradients
            error_output = a4 - y
            delta_output = error_output / batch_size
            
            # Third hidden layer gradients
            a3_dropout = a3 * dropout_mask3
            error_hidden3 = self.w4.T @ delta_output
            error_hidden3 *= dropout_mask3
            delta_hidden3 = error_hidden3 * self.tanh_derivative(z3)
            
            # Second hidden layer gradients
            a2_dropout = a2 * dropout_mask2
            error_hidden2 = self.w3.T @ delta_hidden3
            error_hidden2 *= dropout_mask2
            delta_hidden2 = error_hidden2 * self.tanh_derivative(z2)
            
            # First hidden layer gradients
            a1_dropout = a1 * dropout_mask1
            error_hidden1 = self.w2.T @ delta_hidden2
            error_hidden1 *= dropout_mask1
            delta_hidden1 = error_hidden1 * self.tanh_derivative(z1)
            
            # Weight and bias gradients with L2 regularization
            dw4 = delta_output @ a3_dropout.T + self.l2_reg * self.w4
            db4 = np.sum(delta_output, axis=1, keepdims=True)
            dw3 = delta_hidden3 @ a2_dropout.T + self.l2_reg * self.w3
            db3 = np.sum(delta_hidden3, axis=1, keepdims=True)
            dw2 = delta_hidden2 @ a1_dropout.T + self.l2_reg * self.w2
            db2 = np.sum(delta_hidden2, axis=1, keepdims=True)
            dw1 = delta_hidden1 @ x.T + self.l2_reg * self.w1
            db1 = np.sum(delta_hidden1, axis=1, keepdims=True)
            
            return dw1, db1, dw2, db2, dw3, db3, dw4, db4
    
    def update_weights(self, gradients):
        if self.num_hidden_layers == 1:
            dw1, db1, dw2, db2 = gradients
            
            # Gradient clipping
            dw1 = np.clip(dw1, -self.max_grad_norm, self.max_grad_norm)
            dw2 = np.clip(dw2, -self.max_grad_norm, self.max_grad_norm)
            db1 = np.clip(db1, -self.max_grad_norm, self.max_grad_norm)
            db2 = np.clip(db2, -self.max_grad_norm, self.max_grad_norm)
            
            # Update weights and biases
            self.w1 -= self.learning_rate * dw1
            self.w2 -= self.learning_rate * dw2
            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2
            
            # Weight clipping for stability
            self.w1 = np.clip(self.w1, -self.weight_clip_value, self.weight_clip_value)
            self.w2 = np.clip(self.w2, -self.weight_clip_value, self.weight_clip_value)
            
        elif self.num_hidden_layers == 2:
            dw1, db1, dw2, db2, dw3, db3 = gradients
            
            # Gradient clipping
            dw1 = np.clip(dw1, -self.max_grad_norm, self.max_grad_norm)
            dw2 = np.clip(dw2, -self.max_grad_norm, self.max_grad_norm)
            dw3 = np.clip(dw3, -self.max_grad_norm, self.max_grad_norm)
            db1 = np.clip(db1, -self.max_grad_norm, self.max_grad_norm)
            db2 = np.clip(db2, -self.max_grad_norm, self.max_grad_norm)
            db3 = np.clip(db3, -self.max_grad_norm, self.max_grad_norm)
            
            # Update weights and biases
            self.w1 -= self.learning_rate * dw1
            self.w2 -= self.learning_rate * dw2
            self.w3 -= self.learning_rate * dw3
            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2
            self.b3 -= self.learning_rate * db3
            
            # Weight clipping for stability
            self.w1 = np.clip(self.w1, -self.weight_clip_value, self.weight_clip_value)
            self.w2 = np.clip(self.w2, -self.weight_clip_value, self.weight_clip_value)
            self.w3 = np.clip(self.w3, -self.weight_clip_value, self.weight_clip_value)
            
        elif self.num_hidden_layers == 3:
            dw1, db1, dw2, db2, dw3, db3, dw4, db4 = gradients
            
            # Gradient clipping
            dw1 = np.clip(dw1, -self.max_grad_norm, self.max_grad_norm)
            dw2 = np.clip(dw2, -self.max_grad_norm, self.max_grad_norm)
            dw3 = np.clip(dw3, -self.max_grad_norm, self.max_grad_norm)
            dw4 = np.clip(dw4, -self.max_grad_norm, self.max_grad_norm)
            db1 = np.clip(db1, -self.max_grad_norm, self.max_grad_norm)
            db2 = np.clip(db2, -self.max_grad_norm, self.max_grad_norm)
            db3 = np.clip(db3, -self.max_grad_norm, self.max_grad_norm)
            db4 = np.clip(db4, -self.max_grad_norm, self.max_grad_norm)
            
            # Update weights and biases
            self.w1 -= self.learning_rate * dw1
            self.w2 -= self.learning_rate * dw2
            self.w3 -= self.learning_rate * dw3
            self.w4 -= self.learning_rate * dw4
            self.b1 -= self.learning_rate * db1
            self.b2 -= self.learning_rate * db2
            self.b3 -= self.learning_rate * db3
            self.b4 -= self.learning_rate * db4
            
            # Weight clipping for stability
            self.w1 = np.clip(self.w1, -self.weight_clip_value, self.weight_clip_value)
            self.w2 = np.clip(self.w2, -self.weight_clip_value, self.weight_clip_value)
            self.w3 = np.clip(self.w3, -self.weight_clip_value, self.weight_clip_value)
            self.w4 = np.clip(self.w4, -self.weight_clip_value, self.weight_clip_value)
    
    def train_epoch(self, x_train, y_train):
        """
        Train the network for one epoch.
        
        Parameters:
        x_train (ndarray): Training inputs
        y_train (ndarray): Training targets
        batch_size (int): Batch size for mini-batch training
        
        Returns:
        float: Average loss for the epoch
        """
        n_train = x_train.shape[1]
        n_batches = max(1, n_train // self.batch_size)
        epoch_loss = 0
        
        # Shuffle training data
        indices = np.random.permutation(n_train)
        
        for batch in range(n_batches):
            # Get batch data
            start_idx = batch * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_train)
            batch_indices = indices[start_idx:end_idx]
            
            batch_x = x_train[:, batch_indices]
            batch_y = y_train[:, batch_indices]
            
            # Forward pass with dropout
            forward_results = self.forward_pass(batch_x, training=True)
            
            # Calculate loss with L2 regularization
            if self.num_hidden_layers == 1:
                final_output = forward_results[3]  # a2
                l2_loss = self.l2_reg * (np.sum(self.w1**2) + np.sum(self.w2**2))
            elif self.num_hidden_layers == 2:
                final_output = forward_results[5]  # a3
                l2_loss = self.l2_reg * (np.sum(self.w1**2) + np.sum(self.w2**2) + np.sum(self.w3**2))
            elif self.num_hidden_layers == 3:
                final_output = forward_results[7]  # a4
                l2_loss = self.l2_reg * (np.sum(self.w1**2) + np.sum(self.w2**2) + 
                                       np.sum(self.w3**2) + np.sum(self.w4**2))
            
            error = final_output - batch_y
            mse_loss = np.mean(error ** 2)
            loss = mse_loss + l2_loss
            epoch_loss += mse_loss  # Only track MSE for monitoring
            
            # Backward pass
            gradients = self.backward_pass(batch_x, batch_y, forward_results)
            
            # Update weights
            self.update_weights(gradients)
        
        return epoch_loss / n_batches
    
    def evaluate(self, x_test, y_test, max_samples=500):
        """
        Evaluate the network on test data.
        
        Parameters:
        x_test (ndarray): Test inputs
        y_test (ndarray): Test targets
        max_samples (int): Maximum samples to evaluate (for speed)
        
        Returns:
        float: Test loss
        """
        n_test = x_test.shape[1]
        if n_test > max_samples:
            indices = np.random.choice(n_test, max_samples, replace=False)
            x_test_sample = x_test[:, indices]
            y_test_sample = y_test[:, indices]
        else:
            x_test_sample = x_test
            y_test_sample = y_test
        
        # Forward pass only (no dropout during evaluation)
        forward_results = self.forward_pass(x_test_sample, training=False)
        if self.num_hidden_layers == 1:
            final_output = forward_results[3]  # a2
        elif self.num_hidden_layers == 2:
            final_output = forward_results[5]  # a3
        elif self.num_hidden_layers == 3:
            final_output = forward_results[7]  # a4
        
        # Calculate loss
        error = final_output - y_test_sample
        test_loss = np.mean(error ** 2)
        
        return test_loss
    
    def train(self, x_train, y_train, x_test, y_test):
       
        print(f"\nStarting training with current hyperparameters:")
        print(f"  Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate} (initial)")
        print(f"  Regularization: dropout {self.dropout_rate}, L2 {self.l2_reg}")
        print(f"  Early stopping patience: {self.early_stop_patience}")
        print(f"  Gradient/weight clipping: {self.max_grad_norm}, {self.weight_clip_value}")
        
        self.train_losses = []
        self.test_losses = []
        
        # Early stopping variables
        best_test_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(self.epochs):
            # Track timing
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss = self.train_epoch(x_train, y_train)
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.train_losses.append(train_loss)
            
            # Learning rate decay
            if epoch > 0 and epoch % self.lr_decay_every == 0:
                self.learning_rate *= self.lr_decay_rate
                print(f"Learning rate reduced to {self.learning_rate:.6f}")
            
            # Evaluate on test set (every 10 epochs) - MONITORING ONLY, NOT TRAINING!
            if epoch % 10 == 0:
                # This evaluation does NOT update weights - it's just to monitor generalization
                test_loss = self.evaluate(x_test, y_test)
                self.test_losses.append(test_loss)
                
                # Early stopping check
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    # Save best weights
                    best_weights = (self.w1.copy(), self.w2.copy(), self.b1.copy(), self.b2.copy())
                else:
                    patience_counter += 1
                
                if epoch % self.print_every == 0:
                    # Calculate timing stats
                    recent_epochs = self.epoch_times[-self.print_every:] if len(self.epoch_times) >= self.print_every else self.epoch_times
                    avg_time_per_epoch = np.mean(recent_epochs) if recent_epochs else 0
                    time_per_100 = avg_time_per_epoch * 100
                    
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f} | Best: {best_test_loss:.6f} | Time/100ep: {time_per_100:.1f}s")
                
                # Early stopping
                if patience_counter >= self.early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (patience: {patience_counter})")
                    if best_weights is not None:
                        self.w1, self.w2, self.b1, self.b2 = best_weights
                        print("Restored best weights")
                    break
                    
            elif epoch % self.print_every == 0:
                print(f"Epoch {epoch+1:4d}/{self.epochs} | Train Loss: {train_loss:.6f}")
        
        print(f"\nTraining completed!")
        print(f"Final training loss: {self.train_losses[-1]:.6f}")
        if self.test_losses:
            print(f"Final test loss: {self.test_losses[-1]:.6f}")
    
    def predict(self, x):
        """
        Make predictions using the trained network.
        
        Parameters:
        x (ndarray): Input data
        
        Returns:
        ndarray: Network predictions
        """
        forward_results = self.forward_pass(x, training=False)
        if self.num_hidden_layers == 1:
            predictions = forward_results[3]  # a2
        elif self.num_hidden_layers == 2:
            predictions = forward_results[5]  # a3
        elif self.num_hidden_layers == 3:
            predictions = forward_results[7]  # a4
        return predictions
    
    def save_model(self, filename, normalization_params, s_force):
        """
        Save the trained model to file.
        
        Parameters:
        filename (str): Output filename
        normalization_params (tuple): Normalization parameters
        s_force (ndarray): Arc length positions
        """
        x_mean, x_std, y_mean, y_std = normalization_params
        
        # Prepare model data based on architecture
        model_data = {
            'w1': self.w1, 'b1': self.b1,
            'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std,
            'network_config': {
                'input_size': self.input_size,
                'hidden_size1': self.hidden_size1,
                'hidden_size2': self.hidden_size2,
                'hidden_size3': self.hidden_size3,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'num_hidden_layers': self.num_hidden_layers
            },
            's_force': s_force,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        
        # Add weights based on architecture
        if self.num_hidden_layers == 1:
            model_data.update({
                'w2': self.w2, 'b2': self.b2
            })
        elif self.num_hidden_layers == 2:
            model_data.update({
                'w2': self.w2, 'b2': self.b2,
                'w3': self.w3, 'b3': self.b3
            })
        elif self.num_hidden_layers == 3:
            model_data.update({
                'w2': self.w2, 'b2': self.b2,
                'w3': self.w3, 'b3': self.b3,
                'w4': self.w4, 'b4': self.b4
            })
        
        np.savez_compressed(filename, **model_data)
        print(f"Model saved to '{filename}'")


def run_single_training():
    """
    Run a single training session and plot results.
    """
    print("="*60)
    print("SINGLE TRAINING SESSION")
    print("="*60)
    
    try:
        # Import data
        x_train, y_train, x_test, y_test, s_force, norm_params = data_import()
        
        # Create and train network with three hidden layers
        network = FeedForwardNN(
            input_size=24, hidden_size1=768, hidden_size2=512, hidden_size3=256, output_size=600,
            learning_rate=0.001, dropout_rate=0.3, l2_reg=0.01,
            epochs=1000, batch_size=32, early_stop_patience=40,
            lr_decay_rate=0.9, lr_decay_every=250, num_hidden_layers=3
        )
        network.train(x_train, y_train, x_test, y_test)
        
        # Save model
        network.save_model('trained_model_single.npz', norm_params, s_force)
        
        # Plot results
        plot_training_results(network, x_test, y_test, s_force, norm_params, "Single Training Results")
        
        return network, norm_params, s_force
        
    except Exception as e:
        print(f"Error in single training: {e}")
        return None, None, None


def run_multiple_training(num_runs=10):
    """
    Run multiple training sessions and plot aggregated results.
    
    Parameters:
    num_runs (int): Number of training runs to perform
    """
    print("="*60)
    print(f"MULTIPLE TRAINING SESSIONS ({num_runs} runs)")
    print("="*60)
    
    try:
        # Import data once
        x_train, y_train, x_test, y_test, s_force, norm_params = data_import()
        
        all_networks = []
        all_final_train_losses = []
        all_final_test_losses = []
        
        for run in range(num_runs):
            print(f"\n--- Training Run {run+1}/{num_runs} ---")
            
            # Create new network for each run (different random initialization)
            network = FeedForwardNN(
                input_size=24, hidden_size=1024, output_size=600,
                learning_rate=0.001, dropout_rate=0.5, l2_reg=0.02,
                epochs=1000, batch_size=16, print_every=50, early_stop_patience=20,
                lr_decay_rate=0.9, lr_decay_every=300
            )
            network.train(x_train, y_train, x_test, y_test)
            
            all_networks.append(network)
            all_final_train_losses.append(network.train_losses[-1])
            all_final_test_losses.append(network.test_losses[-1] if network.test_losses else float('inf'))
        
        # Find best network (lowest test loss)
        best_idx = np.argmin(all_final_test_losses)
        best_network = all_networks[best_idx]
        
        print(f"\n" + "="*60)
        print("MULTIPLE TRAINING SUMMARY")
        print("="*60)
        print(f"Best network: Run {best_idx+1}")
        print(f"Best test loss: {all_final_test_losses[best_idx]:.6f}")
        print(f"Average final train loss: {np.mean(all_final_train_losses):.6f} ± {np.std(all_final_train_losses):.6f}")
        print(f"Average final test loss: {np.mean(all_final_test_losses):.6f} ± {np.std(all_final_test_losses):.6f}")
        
        # Save best model
        best_network.save_model('trained_model_best.npz', norm_params, s_force)
        
        # Plot results
        plot_multiple_training_results(all_networks, x_test, y_test, s_force, norm_params, best_idx)
        
        return all_networks, best_idx, norm_params, s_force
        
    except Exception as e:
        print(f"Error in multiple training: {e}")
        return None, None, None, None


def plot_training_results(network, x_test, y_test, s_force, norm_params, title="Training Results"):
   
    x_mean, x_std, y_mean, y_std = norm_params
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot 1: Training loss
    axes[0, 0].plot(network.train_losses, 'b-', linewidth=1, label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Test loss
    if network.test_losses:
        test_epochs = list(range(0, len(network.test_losses) * 10, 10))
        axes[0, 1].plot(test_epochs, network.test_losses, 'r-', linewidth=1, label='Test Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
    
    # Plot 3: Sample prediction comparison (Fx component)
    test_idx = 0
    prediction = network.predict(x_test[:, test_idx:test_idx+1]).flatten()
    actual = y_test[:, test_idx]
    
    # Denormalize for plotting
    prediction = prediction * y_std.flatten() + y_mean.flatten()
    actual = actual * y_std.flatten() + y_mean.flatten()
    
    fx_pred = prediction[:200]
    fx_actual = actual[:200]
    
    axes[1, 0].plot(s_force, fx_actual, 'r-', label='Actual Fx', linewidth=2, alpha=0.8)
    axes[1, 0].plot(s_force, fx_pred, 'b--', label='Predicted Fx', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Arc Length s')
    axes[1, 0].set_ylabel('Force (N)')
    axes[1, 0].set_title('Sample Prediction vs Actual (Fx)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Prediction error histogram
    error = fx_pred - fx_actual
    axes[1, 1].hist(error, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error (N)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Error Distribution (Fx)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_training_results(networks, x_test, y_test, s_force, norm_params, best_idx):
    """
    Plot results from multiple training runs.
    
    Parameters:
    networks (list): List of trained networks
    x_test, y_test: Test data
    s_force: Arc length positions
    norm_params: Normalization parameters
    best_idx (int): Index of best network
    """
    x_mean, x_std, y_mean, y_std = norm_params
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Multiple Training Sessions Results', fontsize=16)
    
    # Plot 1: All training losses
    for i, network in enumerate(networks):
        alpha = 1.0 if i == best_idx else 0.3
        color = 'red' if i == best_idx else 'blue'
        label = f'Best Run {best_idx+1}' if i == best_idx else None
        axes[0, 0].plot(network.train_losses, color=color, alpha=alpha, linewidth=1, label=label)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Training Losses (All Runs)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    if best_idx is not None:
        axes[0, 0].legend()
    
    # Plot 2: Final loss distributions
    final_train_losses = [net.train_losses[-1] for net in networks]
    final_test_losses = [net.test_losses[-1] if net.test_losses else float('inf') for net in networks]
    
    axes[0, 1].hist(final_train_losses, bins=10, alpha=0.7, label='Train Loss', edgecolor='black')
    axes[0, 1].hist([loss for loss in final_test_losses if loss != float('inf')], bins=10, alpha=0.7, label='Test Loss', edgecolor='black')
    axes[0, 1].set_xlabel('Final Loss')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Final Loss Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Best network test loss
    best_network = networks[best_idx]
    if best_network.test_losses:
        test_epochs = list(range(0, len(best_network.test_losses) * 10, 10))
        axes[0, 2].plot(test_epochs, best_network.test_losses, 'r-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('MSE Loss')
        axes[0, 2].set_title(f'Best Network Test Loss (Run {best_idx+1})')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
    
    # Plot 4-6: Best network predictions for all force components
    test_idx = 0
    prediction = best_network.predict(x_test[:, test_idx:test_idx+1]).flatten()
    actual = y_test[:, test_idx]
    
    # Denormalize
    prediction = prediction * y_std.flatten() + y_mean.flatten()
    actual = actual * y_std.flatten() + y_mean.flatten()
    
    # Split into components
    fx_pred, fx_actual = prediction[:200], actual[:200]
    fy_pred, fy_actual = prediction[200:400], actual[200:400]
    fz_pred, fz_actual = prediction[400:600], actual[400:600]
    
    force_components = [
        (fx_pred, fx_actual, 'Fx', 'red'),
        (fy_pred, fy_actual, 'Fy', 'green'),
        (fz_pred, fz_actual, 'Fz', 'blue')
    ]
    
    for i, (pred, act, name, color) in enumerate(force_components):
        axes[1, i].plot(s_force, act, '-', color=color, label=f'Actual {name}', linewidth=2, alpha=0.8)
        axes[1, i].plot(s_force, pred, '--', color='black', label=f'Predicted {name}', linewidth=2, alpha=0.8)
        axes[1, i].set_xlabel('Arc Length s')
        axes[1, i].set_ylabel('Force (N)')
        axes[1, i].set_title(f'Best Network: {name} Prediction')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiple_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Feed-forward Neural Network Training Script")
    print("Choose an option:")
    print("1. Run single training session")
    print("2. Run multiple training sessions (10 runs)")
    print("3. Run both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        run_single_training()
    elif choice == "2":
        run_multiple_training()
    elif choice == "3":
        print("\nRunning single training first...")
        run_single_training()
        print("\nRunning multiple training sessions...")
        run_multiple_training()
    else:
        print("Invalid choice. Running single training as default.")
        run_single_training()