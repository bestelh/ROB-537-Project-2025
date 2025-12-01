import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import random


# ============================================================
# TRANSFORMER MODEL CLASSES
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

class DeformationToForceTransformer(nn.Module):
    """
    Transformer model that maps actuator deformations to distributed forces
    
    Input: (batch_size, 12, 3) - 12 actuator tokens with [initial, deformed, difference]
    Output: (batch_size, 192, 3) - 192 position tokens with [fx, fy, fz]
    """
    
    def __init__(self, 
                 input_dim=3,           # Features per actuator token
                 output_dim=3,          # Force components per position
                 output_seq_len=64,     # Output sequence length (force positions)
                 d_model=256,           # Model dimension
                 nhead=8,               # Number of attention heads
                 num_encoder_layers=6,  # Number of encoder layers
                 num_decoder_layers=6,  # Number of decoder layers
                 dim_feedforward=1024,  # FFN dimension
                 dropout=0.1,           # Dropout rate
                 max_seq_len=512):      # Maximum sequence length
        
        super().__init__()
        
        # Model parameters
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.src_len = 12   # 12 actuators
        self.tgt_len = output_seq_len  # Actual force positions from data
        
        # Input embeddings
        self.src_embed = nn.Linear(input_dim, d_model)
        self.tgt_embed = nn.Linear(output_dim, d_model)
        
        # Positional encodings
        self.src_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first for better performance
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Learnable position queries for force positions
        self.position_queries = nn.Parameter(torch.randn(output_seq_len, d_model))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        """
        Forward pass
        
        Args:
            src: (batch_size, 12, 3) - actuator deformation data
        
        Returns:
            (batch_size, 192, 3) - predicted distributed forces
        """
        batch_size = src.size(0)
        
        # Embed and encode source (actuator data)
        src_embedded = self.src_embed(src)  # (batch_size, 12, d_model)
        src_embedded = self.src_pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)  # Apply pos encoding
        
        # Encode source through transformer encoder
        memory = self.transformer.encoder(src_embedded)  # (batch_size, 12, d_model)
        
        # Always use learnable position queries (simpler and more stable)
        # This treats force prediction as a parallel regression task rather than autoregressive
        position_queries = self.position_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, 192, d_model)
        position_queries = self.tgt_pos_encoding(position_queries.transpose(0, 1)).transpose(0, 1)  # Apply pos encoding
        
        # Decode without causal masking (parallel decoding)
        output = self.transformer.decoder(position_queries, memory)
        
        # Project to output dimension (already in batch_first format)
        output = self.output_proj(output)  # (batch_size, 192, 3)
        
        return output
    

    
    def predict(self, src):
        """Inference method"""
        self.eval()
        with torch.no_grad():
            return self.forward(src)

class ForceEstimatorTrainer:
    """Training class for the DeformationToForceTransformer"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        print(f"ForceEstimatorTrainer initialized with device: {self.device}")
    
    def train_model(self, x_train, y_train, 
                   batch_size=32,
                   epochs=100,
                   learning_rate=1e-4,
                   weight_decay=1e-5,
                   val_split=0.1,
                   patience=15,
                   gradient_clip=1.0,
                   save_path='best_transformer.pth'):
        """
        Train the transformer model
        
        Args:
            x_train: (n_samples, 12, 3) - actuator data
            y_train: (n_samples, 192, 3) - force data
            batch_size: Training batch size
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
            val_split: Validation split fraction
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            save_path: Path to save best model
        """
        
        # Convert to tensors
        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Split into train/validation
        n_samples = x_train.size(0)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        # Random split
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        x_train_split = x_train[train_indices]
        y_train_split = y_train[train_indices]
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(x_train_split, y_train_split)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5
        )
        criterion = nn.MSELoss()
        
        # Training loop with timing
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        epoch_times = []
        
        print(f"Training transformer model for {epochs} epochs...")
        print(f"Train samples: {n_train}, Val samples: {n_val}")
        print(f"Device: {self.device}")
        print(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
        
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        print(f"\nStarted training at: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            print(f"\nEpoch {epoch+1:3d}/{epochs} - Training...", end="", flush=True)
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass (don't pass target during training for parallel prediction)
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Show progress every 10 batches
                if (i + 1) % max(1, len(train_loader) // 10) == 0:
                    print(f"\rEpoch {epoch+1:3d}/{epochs} - Training... [{i+1:3d}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}", end="", flush=True)
            
            train_loss /= train_batches
            
            # Validation
            print(f"\rEpoch {epoch+1:3d}/{epochs} - Validating...", end="", flush=True)
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    predictions = self.model(batch_x)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            
            # Calculate timing
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Estimate remaining time
            avg_epoch_time = np.mean(epoch_times[-5:])  # Average of last 5 epochs
            remaining_epochs = epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            lr_changed = new_lr != old_lr
            
            # Early stopping and progress display
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                status = "✓ NEW BEST"
            else:
                patience_counter += 1
                status = f"({patience_counter}/{patience})"
            
            # Print comprehensive epoch summary
            print(f"\rEpoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s | ETA: {eta.strftime('%H:%M:%S')} | "
                  f"{status}")
            
            if lr_changed:
                print(f"         LR reduced: {old_lr:.2e} → {new_lr:.2e}")
            
            # GPU memory monitoring (every 10 epochs)
            if self.device.type == 'cuda' and (epoch + 1) % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1e9
                cached = torch.cuda.memory_reserved() / 1e9
                print(f"         GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
            
            # Early stopping check
            if patience_counter >= patience:
                total_time = time.time() - start_time
                print(f"\nEarly stopping after {epoch+1} epochs")
                print(f"Total training time: {total_time/60:.1f} minutes")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        
        # Final summary
        total_time = time.time() - start_time
        final_epoch = len(self.train_losses)
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print(f"Total epochs: {final_epoch}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Average time per epoch: {avg_epoch_time:.1f} seconds")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        if self.device.type == 'cuda':
            print(f"Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print("=" * 80)
        
        
        # Show training examples at the end
        self._show_training_examples(x_train, y_train)
        
        return self.train_losses, self.val_losses
    
    def _show_training_examples(self, x_train, y_train):
        """Show how well the model fits to two random training examples"""
        import random
        
        print("\n" + "="*50)
        print("TRAINING DATA FIT EXAMPLES")
        print("="*50)
        
        # Convert to tensor if numpy
        if isinstance(x_train, np.ndarray):
            x_tensor = torch.FloatTensor(x_train).to(self.device)
            y_tensor = torch.FloatTensor(y_train).to(self.device)
        else:
            x_tensor = x_train
            y_tensor = y_train
        
        # Select two random examples
        n_samples = x_tensor.shape[0]
        sample_indices = random.sample(range(n_samples), 2)
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions for the two samples
            x_examples = x_tensor[sample_indices]
            y_examples = y_tensor[sample_indices]
            predictions = self.model(x_examples)
            
            # Convert back to numpy for plotting
            x_np = x_examples.cpu().numpy()
            y_np = y_examples.cpu().numpy()
            pred_np = predictions.cpu().numpy()
        
        # Create plots
        fig, axes = plt.subplots(2, 6, figsize=(18, 8))
        fig.suptitle('Training Data Fit - Two Random Examples', fontsize=16)
        
        for sample_idx in range(2):
            sample_num = sample_indices[sample_idx] + 1
            
            # Calculate metrics for this sample
            sample_mse = np.mean((pred_np[sample_idx] - y_np[sample_idx])**2)
            sample_mae = np.mean(np.abs(pred_np[sample_idx] - y_np[sample_idx]))
            
            print(f"\nTraining Sample #{sample_num}:")
            print(f"  MSE: {sample_mse:.6f}")
            print(f"  MAE: {sample_mae:.6f}")
            
            for j, component in enumerate(['Fx', 'Fy', 'Fz']):
                ax = axes[sample_idx, j]
                
                # Plot prediction vs true
                ax.plot(pred_np[sample_idx, :, j], label='Predicted', linewidth=2, color='blue', alpha=0.8)
                ax.plot(y_np[sample_idx, :, j], label='True', linewidth=2, color='red', alpha=0.8)
                ax.set_title(f'Sample #{sample_num} - {component}')
                ax.set_xlabel('Position')
                ax.set_ylabel(f'{component} (N)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Plot error
                ax_error = axes[sample_idx, j + 3]
                error = pred_np[sample_idx, :, j] - y_np[sample_idx, :, j]
                ax_error.plot(error, color='red', linewidth=2)
                ax_error.set_title(f'{component} Error (MAE: {np.mean(np.abs(error)):.4f})')
                ax_error.set_xlabel('Position')
                ax_error.set_ylabel('Error (N)')
                ax_error.grid(True, alpha=0.3)
                ax_error.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print("\nThese examples show how well your trained model fits the training data.")
        print("Lower errors indicate better training data fit.")
    
    def save_model_with_config(self, save_path='complete_transformer_model.pth'):
        """Save model with its configuration for easy loading"""
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'output_seq_len': self.model.tgt_len,
                'd_model': self.model.d_model,
                'nhead': self.model.transformer.nhead,
                'num_encoder_layers': self.model.transformer.encoder.num_layers,
                'num_decoder_layers': self.model.transformer.decoder.num_layers,
                'dim_feedforward': self.model.transformer.encoder.layers[0].linear1.out_features,
                'dropout': 0.1  # Default, could be extracted if needed
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(model_data, save_path)
        print(f"Complete model saved to {save_path}")
        return save_path
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def evaluate_model(self, x_test, y_test, n_samples=5):
        """Evaluate model and show sample predictions"""
        x_test = torch.FloatTensor(x_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
            mse_loss = nn.MSELoss()(predictions, y_test)
            mae_loss = nn.L1Loss()(predictions, y_test)
        
        print(f"Test MSE: {mse_loss.item():.6f}")
        print(f"Test MAE: {mae_loss.item():.6f}")
        
        # Show sample predictions
        for i in range(min(n_samples, x_test.size(0))):
            pred = predictions[i].cpu().numpy()
            true = y_test[i].cpu().numpy()
            
            plt.figure(figsize=(15, 5))
            for j, component in enumerate(['Fx', 'Fy', 'Fz']):
                plt.subplot(1, 3, j+1)
                plt.plot(true[:, j], label='True', alpha=0.7)
                plt.plot(pred[:, j], label='Predicted', alpha=0.7)
                plt.title(f'Sample {i+1} - {component}')
                plt.xlabel('Position')
                plt.ylabel('Force')
                plt.legend()
                plt.grid(True)
            plt.tight_layout()
            plt.show()

# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(npz_file_path="11_30_training.npz"):
    """
    Load training data from .npz file
    
    Returns:
        x_train: (n_samples, 12, 3) - 12 actuator tokens with [initial, deformed, difference]
        y_train: (n_samples, 192, 3) - 192 position tokens with [fx, fy, fz]
    """
    from pathlib import Path
    
    # Try multiple possible file locations
    script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    possible_files = [
        npz_file_path,
        script_dir / npz_file_path,
        script_dir.parent / npz_file_path,
        Path.cwd() / npz_file_path,
    ]
    
    actual_file_path = None
    for filepath in possible_files:
        if Path(filepath).exists():
            actual_file_path = str(filepath)
            break
    
    if actual_file_path is None:
        raise FileNotFoundError(f"Could not find {npz_file_path} in any of the searched locations")
    
    data = np.load(actual_file_path)
    x_train = data["x_train"]  # Shape: (n_samples, 12, 3)
    y_train = data["y_train"]  # Shape: (n_samples, ?, 3)
    
    print(f"Loaded training data:")
    print(f"  Input shape: {x_train.shape} (samples, actuator_tokens, features)")
    print(f"  Output shape: {y_train.shape} (samples, position_tokens, force_xyz)")
    
    # Get actual output sequence length
    actual_output_len = y_train.shape[1]
    
    return x_train, y_train, actual_output_len




if __name__ == "__main__":
    # Check GPU availability
    print("Checking GPU availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = torch.device('cuda')
    else:
        print("No GPU found, using CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load training data
    x_train, y_train, actual_output_len = load_training_data("11_30_training.npz")
    
    # Print data shapes and some statistics
    print(f"\nTraining data statistics:")
    print(f"  Input (actuators): {x_train.shape}")
    print(f"  Output (forces): {y_train.shape}")
    print(f"  Input range: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"  Output range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Create transformer model with actual data dimensions
    print(f"\nCreating DeformationToForceTransformer with {actual_output_len} output positions...")
    model = DeformationToForceTransformer(
        input_dim=3,                    # [initial, deformed, difference]
        output_dim=3,                   # [fx, fy, fz]
        output_seq_len=actual_output_len,  # Actual sequence length from data
        d_model=256,                    # Model dimension
        nhead=8,                        # Attention heads
        num_encoder_layers=6,           # Encoder layers
        num_decoder_layers=6,           # Decoder layers
        dim_feedforward=1024,           # FFN dimension
        dropout=0.1                     # Dropout rate
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer with explicit device
    trainer = ForceEstimatorTrainer(model, device=device)
    
    # Train the model
    print("\nStarting training...")
    # Use larger batch size for GPU
    batch_size = 64 if device.type == 'cuda' else 32
    print(f"Using batch size: {batch_size}")
    
    train_losses, val_losses = trainer.train_model(
        x_train, y_train,
        batch_size=batch_size,
        epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-5,
        val_split=0.1,
        patience=15,
        save_path='deformation_to_force_transformer.pth'
    )
    
    # Save complete model with configuration
    model_path = trainer.save_model_with_config('complete_transformer_model.pth')
    print(f"Complete model exported to: {model_path}")
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Evaluate on a subset of training data (you could use separate test data)
    print("\nEvaluating model...")
    n_eval = min(1000, x_train.shape[0])
    trainer.evaluate_model(x_train[:n_eval], y_train[:n_eval], n_samples=3)
    
    # Visualize a sample prediction
    sample_idx = 0
    print(f"\nDetailed analysis of sample {sample_idx}:")
    
    # Show actuator data
    print("Actuator data:")
    for seg in range(4):
        print(f"  Segment {seg+1}:")
        for act in range(3):
            idx = seg*3 + act
            initial, deformed, diff = x_train[sample_idx, idx, :]
            print(f"    Act {act+1}: {initial:.3f} → {deformed:.3f} (Δ={diff:.3f})")
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        x_sample = torch.FloatTensor(x_train[sample_idx:sample_idx+1]).to(trainer.device)
        pred_forces = model(x_sample).cpu().numpy()[0]  # (192, 3)
        true_forces = y_train[sample_idx]  # (192, 3)
    
    # Compare prediction vs true
    mse = np.mean((pred_forces - true_forces)**2)
    mae = np.mean(np.abs(pred_forces - true_forces))
    print(f"\nPrediction accuracy for sample {sample_idx}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max error: {np.max(np.abs(pred_forces - true_forces)):.6f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    for i, component in enumerate(['Fx', 'Fy', 'Fz']):
        plt.subplot(2, 3, i+1)
        plt.plot(true_forces[:, i], label='True', linewidth=2)
        plt.plot(pred_forces[:, i], label='Predicted', linewidth=2, alpha=0.7)
        plt.title(f'{component} - Full Range')
        plt.xlabel('Position')
        plt.ylabel(f'{component} (N)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, i+4)
        error = pred_forces[:, i] - true_forces[:, i]
        plt.plot(error, color='red', linewidth=2)
        plt.title(f'{component} - Prediction Error')
        plt.xlabel('Position')
        plt.ylabel('Error (N)')
        plt.grid(True)
    
    plt.suptitle(f'Force Prediction vs Ground Truth - Sample {sample_idx}')
    plt.tight_layout()
    plt.show()