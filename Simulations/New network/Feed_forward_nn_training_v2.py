
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


def data_import(npz_file_path="simulation_results_all_11-19.npz", train_split=0.5):
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
        # Input: All actuator data for segments 1-4 (24 features total)
        # L1_input, L1_def, L2_input, L2_def, L3_input, L3_def for each segment
        inputs = [
            sim['L1_input_1'], sim['L1_def_1'], sim['L2_input_1'], sim['L2_def_1'], sim['L3_input_1'], sim['L3_def_1'],
            sim['L1_input_2'], sim['L1_def_2'], sim['L2_input_2'], sim['L2_def_2'], sim['L3_input_2'], sim['L3_def_2'],
            sim['L1_input_3'], sim['L1_def_3'], sim['L2_input_3'], sim['L2_def_3'], sim['L3_input_3'], sim['L3_def_3'],
            sim['L1_input_4'], sim['L1_def_4'], sim['L2_input_4'], sim['L2_def_4'], sim['L3_input_4'], sim['L3_def_4']
        ]
        
        # Output: Concatenated force distributions (576 values total)
        forces = np.concatenate([
            sim['f_dist_fx'],  # 192 values - Force X
            sim['f_dist_fy'],  # 192 values - Force Y
            sim['f_dist_fz']   # 192 values - Force Z
        ])
        
        input_features.append(inputs)
        output_forces.append(forces)
    
    X = np.array(input_features)  # Shape: (n_samples, 24)
    Y = np.array(output_forces)   # Shape: (n_samples, 576)
    
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
    y_train = Y[:split_idx].T  # Shape: (576, n_train)
    x_test = X[split_idx:].T   # Shape: (24, n_test) - Last 50% for testing
    y_test = Y[split_idx:].T   # Shape: (576, n_test)
    
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
   
    
    def __init__(self, input_size=24, hidden_size1=768, hidden_size2=512, hidden_size3=256, hidden_size4=128, hidden_size5=64,
                 hidden_size6=48, hidden_size7=40, hidden_size8=32, hidden_size9=28, hidden_size10=24,
                 hidden_size11=20, hidden_size12=18, hidden_size13=16, hidden_size14=14, hidden_size15=12, output_size=576, 
                 learning_rate=0.001, dropout_rate=0.3,
                 epochs=200, print_every=50, 
                 early_stop_patience=40, lr_decay_rate=0.9, lr_decay_every=250,
                 max_grad_norm=1.0, weight_clip_value=10.0, num_hidden_layers=3):
        
        # Network architecture
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2 if num_hidden_layers >= 2 else None
        self.hidden_size3 = hidden_size3 if num_hidden_layers >= 3 else None
        self.hidden_size4 = hidden_size4 if num_hidden_layers >= 4 else None
        self.hidden_size5 = hidden_size5 if num_hidden_layers >= 5 else None
        self.hidden_size6 = hidden_size6 if num_hidden_layers >= 6 else None
        self.hidden_size7 = hidden_size7 if num_hidden_layers >= 7 else None
        self.hidden_size8 = hidden_size8 if num_hidden_layers >= 8 else None
        self.hidden_size9 = hidden_size9 if num_hidden_layers >= 9 else None
        self.hidden_size10 = hidden_size10 if num_hidden_layers >= 10 else None
        self.hidden_size11 = hidden_size11 if num_hidden_layers >= 11 else None
        self.hidden_size12 = hidden_size12 if num_hidden_layers >= 12 else None
        self.hidden_size13 = hidden_size13 if num_hidden_layers >= 13 else None
        self.hidden_size14 = hidden_size14 if num_hidden_layers >= 14 else None
        self.hidden_size15 = hidden_size15 if num_hidden_layers >= 15 else None
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        
        # Timing tracking
        self.epoch_times = []
        self.last_print_time = None
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store original for reference
        self.epochs = epochs
        self.print_every = print_every
        
        # Regularization parameters
        self.dropout_rate = dropout_rate
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
            
        elif num_hidden_layers == 4:
            # Four hidden layers: input → h1 → h2 → h3 → h4 → output
            self.w2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2.0 / hidden_size1)
            self.b2 = np.zeros((hidden_size2, 1))
            self.w3 = np.random.randn(hidden_size3, hidden_size2) * np.sqrt(2.0 / hidden_size2)
            self.b3 = np.zeros((hidden_size3, 1))
            self.w4 = np.random.randn(hidden_size4, hidden_size3) * np.sqrt(2.0 / hidden_size3)
            self.b4 = np.zeros((hidden_size4, 1))
            self.w5 = np.random.randn(output_size, hidden_size4) * np.sqrt(2.0 / hidden_size4)
            self.b5 = np.zeros((output_size, 1))
            
        elif num_hidden_layers >= 5:
            # Five or more hidden layers
            self.w2 = np.random.randn(hidden_size2, hidden_size1) * np.sqrt(2.0 / hidden_size1)
            self.b2 = np.zeros((hidden_size2, 1))
            self.w3 = np.random.randn(hidden_size3, hidden_size2) * np.sqrt(2.0 / hidden_size2)
            self.b3 = np.zeros((hidden_size3, 1))
            self.w4 = np.random.randn(hidden_size4, hidden_size3) * np.sqrt(2.0 / hidden_size3)
            self.b4 = np.zeros((hidden_size4, 1))
            self.w5 = np.random.randn(hidden_size5, hidden_size4) * np.sqrt(2.0 / hidden_size4)
            self.b5 = np.zeros((hidden_size5, 1))
            
            if num_hidden_layers >= 6:
                self.w6 = np.random.randn(hidden_size6, hidden_size5) * np.sqrt(2.0 / hidden_size5)
                self.b6 = np.zeros((hidden_size6, 1))
            if num_hidden_layers >= 7:
                self.w7 = np.random.randn(hidden_size7, hidden_size6) * np.sqrt(2.0 / hidden_size6)
                self.b7 = np.zeros((hidden_size7, 1))
            if num_hidden_layers >= 8:
                self.w8 = np.random.randn(hidden_size8, hidden_size7) * np.sqrt(2.0 / hidden_size7)
                self.b8 = np.zeros((hidden_size8, 1))
            if num_hidden_layers >= 9:
                self.w9 = np.random.randn(hidden_size9, hidden_size8) * np.sqrt(2.0 / hidden_size8)
                self.b9 = np.zeros((hidden_size9, 1))
            if num_hidden_layers >= 10:
                self.w10 = np.random.randn(hidden_size10, hidden_size9) * np.sqrt(2.0 / hidden_size9)
                self.b10 = np.zeros((hidden_size10, 1))
            if num_hidden_layers >= 11:
                self.w11 = np.random.randn(hidden_size11, hidden_size10) * np.sqrt(2.0 / hidden_size10)
                self.b11 = np.zeros((hidden_size11, 1))
            if num_hidden_layers >= 12:
                self.w12 = np.random.randn(hidden_size12, hidden_size11) * np.sqrt(2.0 / hidden_size11)
                self.b12 = np.zeros((hidden_size12, 1))
            if num_hidden_layers >= 13:
                self.w13 = np.random.randn(hidden_size13, hidden_size12) * np.sqrt(2.0 / hidden_size12)
                self.b13 = np.zeros((hidden_size13, 1))
            if num_hidden_layers >= 14:
                self.w14 = np.random.randn(hidden_size14, hidden_size13) * np.sqrt(2.0 / hidden_size13)
                self.b14 = np.zeros((hidden_size14, 1))
            if num_hidden_layers >= 15:
                self.w15 = np.random.randn(hidden_size15, hidden_size14) * np.sqrt(2.0 / hidden_size14)
                self.b15 = np.zeros((hidden_size15, 1))
            
            # Output layer - connect from the last hidden layer to output
            if num_hidden_layers == 5:
                self.w_out = np.random.randn(output_size, hidden_size5) * np.sqrt(2.0 / hidden_size5)
            elif num_hidden_layers == 6:
                self.w_out = np.random.randn(output_size, hidden_size6) * np.sqrt(2.0 / hidden_size6)
            elif num_hidden_layers == 7:
                self.w_out = np.random.randn(output_size, hidden_size7) * np.sqrt(2.0 / hidden_size7)
            elif num_hidden_layers == 8:
                self.w_out = np.random.randn(output_size, hidden_size8) * np.sqrt(2.0 / hidden_size8)
            elif num_hidden_layers == 9:
                self.w_out = np.random.randn(output_size, hidden_size9) * np.sqrt(2.0 / hidden_size9)
            elif num_hidden_layers == 10:
                self.w_out = np.random.randn(output_size, hidden_size10) * np.sqrt(2.0 / hidden_size10)
            elif num_hidden_layers == 11:
                self.w_out = np.random.randn(output_size, hidden_size11) * np.sqrt(2.0 / hidden_size11)
            elif num_hidden_layers == 12:
                self.w_out = np.random.randn(output_size, hidden_size12) * np.sqrt(2.0 / hidden_size12)
            elif num_hidden_layers == 13:
                self.w_out = np.random.randn(output_size, hidden_size13) * np.sqrt(2.0 / hidden_size13)
            elif num_hidden_layers == 14:
                self.w_out = np.random.randn(output_size, hidden_size14) * np.sqrt(2.0 / hidden_size14)
            elif num_hidden_layers == 15:
                self.w_out = np.random.randn(output_size, hidden_size15) * np.sqrt(2.0 / hidden_size15)
            
            self.b_out = np.zeros((output_size, 1))
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        
        print(f"Initialized FeedForwardNN with hyperparameters:")
        
        # Display architecture dynamically
        arch_str = f"  Architecture: {input_size}"
        total_params = input_size * hidden_size1  # Input to first hidden layer
        
        # Add all hidden layers
        for layer_num in range(1, num_hidden_layers + 1):
            hidden_size = getattr(self, f'hidden_size{layer_num}')
            arch_str += f" → {hidden_size}"
            
            # Calculate parameters for this layer
            if layer_num > 1:
                prev_hidden_size = getattr(self, f'hidden_size{layer_num-1}')
                total_params += prev_hidden_size * hidden_size
        
        # Add output layer
        arch_str += f" → {output_size}"
        last_hidden_size = getattr(self, f'hidden_size{num_hidden_layers}')
        total_params += last_hidden_size * output_size
        
        print(arch_str)
        print(f"  Total parameters: ~{total_params:,}")
        print(f"  Learning rate: {learning_rate} (decay: {lr_decay_rate} every {lr_decay_every} epochs)")
        print(f"  Training: {epochs} epochs")
        print(f"  Regularization: dropout {dropout_rate}")
        print(f"  Early stopping: patience {early_stop_patience}")
    
    def sigmoid_activation(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid activation"""
        s = self.sigmoid_activation(x)
        return s * (1 - s)
    
    def tanh_activation(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Derivative of tanh activation"""
        t = np.tanh(x)
        return 1 - t * t
    
    def relu_activation(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation"""
        return (x > 0).astype(float)
    
    def linear_activation(self, x):
        """Linear activation function (for output layer)"""
        return x
    
    def swish_activation(self, x):
        """Swish activation function: x * sigmoid(x)"""
        # Clip x to prevent overflow in sigmoid
        x_clipped = np.clip(x, -500, 500)
        sigmoid_x = 1 / (1 + np.exp(-x_clipped))
        return x * sigmoid_x
    
    def swish_derivative(self, x):
        """Derivative of Swish activation"""
        x_clipped = np.clip(x, -500, 500)
        sigmoid_x = 1 / (1 + np.exp(-x_clipped))
        return sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    
    def dropout_mask(self, shape, training=True):# dropout mask randomly turns off neurons during training
        """Generate dropout mask during training"""
        if not training or self.dropout_rate == 0:
            return np.ones(shape)
        mask = np.random.binomial(1, 1 - self.dropout_rate, shape) / (1 - self.dropout_rate)
        return mask
    
    def forward_pass(self, x, training=True):
        """Dynamic forward pass that handles any number of hidden layers"""
        
        # Store all layer outputs and dropout masks
        z_layers = []
        a_layers = []
        dropout_masks = []
        
        # First hidden layer
        z1 = self.w1 @ x + self.b1
        a1 = self.tanh_activation(z1)
        dropout_mask1 = self.dropout_mask(a1.shape, training)
        a1_dropout = a1 * dropout_mask1
        
        z_layers.append(z1)
        a_layers.append(a1)
        dropout_masks.append(dropout_mask1)
        
        current_input = a1_dropout
        
        # Process remaining hidden layers dynamically
        for layer_num in range(2, self.num_hidden_layers + 1):
            w = getattr(self, f'w{layer_num}')
            b = getattr(self, f'b{layer_num}')
            
            z = w @ current_input + b
            a = self.tanh_activation(z)
            dropout_mask = self.dropout_mask(a.shape, training)
            a_dropout = a * dropout_mask
            
            z_layers.append(z)
            a_layers.append(a)
            dropout_masks.append(dropout_mask)
            
            current_input = a_dropout
        
        # Output layer
        z_out = self.w_out @ current_input + self.b_out
        a_out = self.linear_activation(z_out)
        
        z_layers.append(z_out)
        a_layers.append(a_out)
        
        return z_layers, a_layers, dropout_masks
    
    def backward_pass(self, x, y, forward_results):
        """Dynamic backward pass that handles any number of hidden layers"""
        z_layers, a_layers, dropout_masks = forward_results
        n_samples = x.shape[1]
        
        # Initialize gradients storage
        weight_gradients = {}
        bias_gradients = {}
        
        # Output layer gradients (last layer is always output)
        error_output = a_layers[-1] - y
        delta_current = error_output / n_samples
        
        # Output layer weight and bias gradients
        if self.num_hidden_layers > 0:
            a_prev_dropout = a_layers[-2] * dropout_masks[-1]  # Last hidden layer with dropout
            weight_gradients['w_out'] = delta_current @ a_prev_dropout.T
        else:
            weight_gradients['w_out'] = delta_current @ x.T
        bias_gradients['b_out'] = np.sum(delta_current, axis=1, keepdims=True)
        
        # Backpropagate through hidden layers (in reverse order)
        for layer_idx in range(self.num_hidden_layers, 0, -1):
            if layer_idx == self.num_hidden_layers:
                # Last hidden layer connects to output
                w_next = self.w_out
            else:
                # Hidden layers connect to next hidden layer
                w_next = getattr(self, f'w{layer_idx + 1}')
            
            # Compute error and delta for current hidden layer
            error_hidden = w_next.T @ delta_current
            error_hidden *= dropout_masks[layer_idx - 1]  # Apply dropout mask
            delta_current = error_hidden * self.tanh_derivative(z_layers[layer_idx - 1])
            
            # Compute weight and bias gradients
            if layer_idx == 1:
                # First hidden layer connects to input
                input_to_layer = x
            else:
                # Other hidden layers connect to previous hidden layer
                input_to_layer = a_layers[layer_idx - 2] * dropout_masks[layer_idx - 2]
            
            weight_gradients[f'w{layer_idx}'] = delta_current @ input_to_layer.T
            bias_gradients[f'b{layer_idx}'] = np.sum(delta_current, axis=1, keepdims=True)
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, gradients):
        """Dynamic weight update that handles any number of hidden layers"""
        weight_gradients, bias_gradients = gradients
        
        # Update all hidden layer weights and biases
        for layer_num in range(1, self.num_hidden_layers + 1):
            w_key = f'w{layer_num}'
            b_key = f'b{layer_num}'
            
            if w_key in weight_gradients:
                # Gradient clipping
                dw = np.clip(weight_gradients[w_key], -self.max_grad_norm, self.max_grad_norm)
                db = np.clip(bias_gradients[b_key], -self.max_grad_norm, self.max_grad_norm)
                
                # Update weights and biases
                w_attr = getattr(self, w_key)
                b_attr = getattr(self, b_key)
                setattr(self, w_key, w_attr - self.learning_rate * dw)
                setattr(self, b_key, b_attr - self.learning_rate * db)
                
                # Weight clipping for stability
                w_updated = getattr(self, w_key)
                setattr(self, w_key, np.clip(w_updated, -self.weight_clip_value, self.weight_clip_value))
        
        # Update output layer weights and biases
        if 'w_out' in weight_gradients:
            dw_out = np.clip(weight_gradients['w_out'], -self.max_grad_norm, self.max_grad_norm)
            db_out = np.clip(bias_gradients['b_out'], -self.max_grad_norm, self.max_grad_norm)
            
            self.w_out -= self.learning_rate * dw_out
            self.b_out -= self.learning_rate * db_out
            
            # Weight clipping for stability
            self.w_out = np.clip(self.w_out, -self.weight_clip_value, self.weight_clip_value)
    
    def train_epoch(self, x_train, y_train, batch_size=32):
        """
        Train the network for one epoch using mini-batches.
        
        Parameters:
        x_train (ndarray): Training inputs
        y_train (ndarray): Training targets
        batch_size (int): Size of mini-batches
        
        Returns:
        float: Average loss for the epoch
        """
        # Shuffle training data
        n_train = x_train.shape[1]
        indices = np.random.permutation(n_train)
        x_shuffled = x_train[:, indices]
        y_shuffled = y_train[:, indices]
        
        epoch_losses = []
        
        # Process data in batches
        for batch_start in range(0, n_train, batch_size):
            batch_end = min(batch_start + batch_size, n_train)
            
            # Extract batch
            x_batch = x_shuffled[:, batch_start:batch_end]
            y_batch = y_shuffled[:, batch_start:batch_end]
            
            # Forward pass with dropout
            forward_results = self.forward_pass(x_batch, training=True)
            z_layers, a_layers, dropout_masks = forward_results
            
            # Calculate loss
            final_output = a_layers[-1]  # Last layer is always the output
            error = final_output - y_batch
            batch_loss = np.mean(error ** 2)
            epoch_losses.append(batch_loss)
            
            # Backward pass
            gradients = self.backward_pass(x_batch, y_batch, forward_results)
            
            # Update weights
            self.update_weights(gradients)
        
        # Return average loss across all batches
        return np.mean(epoch_losses)
    
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
        z_layers, a_layers, dropout_masks = self.forward_pass(x_test_sample, training=False)
        final_output = a_layers[-1]  # Last layer is always the output
        
        # Calculate loss
        error = final_output - y_test_sample
        test_loss = np.mean(error ** 2)
        
        return test_loss
    
    def train(self, x_train, y_train, x_test, y_test, batch_size=32):
       
        print(f"\nStarting training with current hyperparameters:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {self.learning_rate} (initial)")
        print(f"  Regularization: dropout {self.dropout_rate}")
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
            
            # Train one epoch with batching
            train_loss = self.train_epoch(x_train, y_train, batch_size=batch_size)
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            self.train_losses.append(train_loss)
            
            # Learning rate decay (commented out)
            # if epoch > 0 and epoch % self.lr_decay_every == 0:
            #     self.learning_rate *= self.lr_decay_rate
            #     print(f"Learning rate reduced to {self.learning_rate:.6f}")
            
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
        z_layers, a_layers, dropout_masks = self.forward_pass(x, training=False)
        predictions = a_layers[-1]  # Last layer is always the output
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
        
        # Prepare model data with base configuration
        model_data = {
            'w1': self.w1, 'b1': self.b1,
            'w_out': self.w_out, 'b_out': self.b_out,
            'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std,
            'network_config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate,
                'num_hidden_layers': self.num_hidden_layers
            },
            's_force': s_force,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses
        }
        
        # Add all hidden layer sizes to config
        for layer_num in range(1, self.num_hidden_layers + 1):
            model_data['network_config'][f'hidden_size{layer_num}'] = getattr(self, f'hidden_size{layer_num}')
        
        # Add all hidden layer weights dynamically
        for layer_num in range(2, self.num_hidden_layers + 1):
            w_name = f'w{layer_num}'
            b_name = f'b{layer_num}'
            if hasattr(self, w_name):
                model_data[w_name] = getattr(self, w_name)
                model_data[b_name] = getattr(self, b_name)
        
        np.savez_compressed(filename, **model_data)
        print(f"Model saved to '{filename}'")


def run_single_training():
    """
    Run a single training session and plot results.
    """
    print("="*60)
    print("SINGLE TRAINING SESSION - November 19th Data")
    print("="*60)
    
    try:
        # Import data from November 19th simulation results
        x_train, y_train, x_test, y_test, s_force, norm_params = data_import("simulation_results_all_11-19.npz")
        
        # Create and train network with fifteen hidden layers
        network = FeedForwardNN(
            input_size=24, hidden_size1=300, hidden_size2=200, hidden_size3=100, 
            hidden_size4=80, hidden_size5=60, hidden_size6=48, hidden_size7=40,
            hidden_size8=32, hidden_size9=28, hidden_size10=24, hidden_size11=20,
            hidden_size12=18, hidden_size13=16, hidden_size14=14, hidden_size15=12,
            output_size=576, learning_rate=0.001, dropout_rate=0.1,
            epochs=2000, early_stop_patience=40,
            lr_decay_rate=0.9, lr_decay_every=250, num_hidden_layers=5
        )
        network.train(x_train, y_train, x_test, y_test, batch_size=4)
        
        # Save model
        network.save_model('trained_model_single_11-19.npz', norm_params, s_force)
        
        # Plot results
        plot_training_results(network, x_test, y_test, s_force, norm_params, "Single Training Results - Nov 19 Data")
        
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
    print(f"MULTIPLE TRAINING SESSIONS ({num_runs} runs) - November 19th Data")
    print("="*60)
    
    try:
        # Import data from November 19th simulation results
        x_train, y_train, x_test, y_test, s_force, norm_params = data_import("simulation_results_all_11-19.npz")
        
        all_networks = []
        all_final_train_losses = []
        all_final_test_losses = []
        
        for run in range(num_runs):
            print(f"\n--- Training Run {run+1}/{num_runs} ---")
            
            # Create new network for each run (different random initialization)
            network = FeedForwardNN(
                input_size=24, hidden_size1=1024, output_size=576,
                learning_rate=0.001, dropout_rate=0.5,
                epochs=1000, print_every=50, early_stop_patience=20,
                lr_decay_rate=0.9, lr_decay_every=300
            )
            network.train(x_train, y_train, x_test, y_test, batch_size=32)
            
            all_networks.append(network)
            all_final_train_losses.append(network.train_losses[-1])
            all_final_test_losses.append(network.test_losses[-1] if network.test_losses else float('inf'))
        
        # Find best network (lowest test loss)
        best_idx = np.argmin(all_final_test_losses)
        best_network = all_networks[best_idx]
        
        print(f"\n" + "="*60)
        print("MULTIPLE TRAINING SUMMARY - November 19th Data")
        print("="*60)
        print(f"Best network: Run {best_idx+1}")
        print(f"Best test loss: {all_final_test_losses[best_idx]:.6f}")
        print(f"Average final train loss: {np.mean(all_final_train_losses):.6f} ± {np.std(all_final_train_losses):.6f}")
        print(f"Average final test loss: {np.mean(all_final_test_losses):.6f} ± {np.std(all_final_test_losses):.6f}")
        
        # Save best model
        best_network.save_model('trained_model_best_11-19.npz', norm_params, s_force)
        
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
    
    fx_pred = prediction[:192]
    fx_actual = actual[:192]
    
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
    fig.suptitle('Multiple Training Sessions Results - Nov 19 Data', fontsize=16)
    
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
    
    # Split into components (192 values each)
    fx_pred, fx_actual = prediction[:192], actual[:192]
    fy_pred, fy_actual = prediction[192:384], actual[192:384]
    fz_pred, fz_actual = prediction[384:576], actual[384:576]
    
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
    plt.savefig('multiple_training_results_11-19.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Feed-forward Neural Network Training Script - November 19th Data")
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