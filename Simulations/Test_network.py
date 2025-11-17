import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    
    # ***controls***
    hidden_layer_size = 100 # This assignment uses 1 hidden layer
    input_layer_size = 5
    output_layer_size = 2
    epochs = 1000
    n = 0.1

    # **loading in train data**
    data= pd.read_csv('train1.csv').values.astype(float)

    # randomizing the order of the rows of data
    np.random.shuffle(data)
    
    # splitting data
    x = data[:,0:input_layer_size].T
    y = data[:,input_layer_size:input_layer_size+output_layer_size].T

    # weights and biases 
    w1 = np.random.randn(hidden_layer_size, input_layer_size) 
    w2 = np.random.randn(output_layer_size, hidden_layer_size) 

    b1 = np.zeros((hidden_layer_size, 1))
    b2 = np.zeros((output_layer_size, 1))
    
    # arrays to hold values
    num_samples = x.shape[1]
    output = np.zeros((num_samples, 4))
    per_correct = np.zeros((epochs, 1))
    
    def sigmoid(x):

        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    for epoch in range(epochs):

        for i in range(x.shape[1]):
                
            # hidden layer
            z1= w1 @ x[:,i:i+1] + b1
            a1 = sigmoid(z1)

            # output layer
            z2 = w2 @ a1 + b2
            a2 = sigmoid(z2)

            Y = a2  
            
        
            output[i, 0:2] = Y.T 
            
            # Convert predictions to binary and store in columns 2-3
            if output[i, 0] >= 0.5:
                pred_binary = [1, 0]
            else:
                pred_binary = [0, 1]
            
            # Store binary predictions for comparison
            output[i, 2:4] = pred_binary
            
            target = y[:,i:i+1]

            # finding errors and deltas
            e_output_l = target - Y  # error output layer
            delta_output_l = e_output_l * (Y * (1 - Y)) # delta output layer

            e_hidden_l = w2.T @ delta_output_l  # error hidden layer
            delta_hidden_l = e_hidden_l * (a1 * (1 - a1)) # delta hidden layer

            # finding derivatives of error with respect to weights and biases
            dE_dw1 = delta_hidden_l @ x[:,i:i+1].T
            dE_dw2 = delta_output_l @ a1.T

            dE_db1 = delta_hidden_l  
            dE_db2 = delta_output_l  

            # update weights and biases
            w1 = w1 + n * dE_dw1
            w2 = w2 + n * dE_dw2

            b1 = b1 + n * dE_db1
            b2 = b2 + n * dE_db2
           

         # calculate accuracy 
        output_compare = np.column_stack([output, y.T])  
        
        # check if predicted matches target
        accuracy_per_sample = np.zeros(x.shape[1])
        for q in range(x.shape[1]):
            predicted = output_compare[q, 2:4]
            target = output_compare[q, 4:6]
            accuracy_per_sample[q] = np.array_equal(predicted, target)
        
        per_correct[epoch, 0] = np.sum(accuracy_per_sample) / x.shape[1]
        
        # printing progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Percent Correct: {per_correct[epoch, 0]:.4f}")
    
    # Export trained weights and biases
    print(f"\nTraining completed! Final accuracy: {per_correct[-1, 0]:.4f}")
    
    # Save weights and biases to files
    np.save('trained_weights_w1.npy', w1)
    np.save('trained_weights_w2.npy', w2)
    np.save('trained_biases_b1.npy', b1)
    np.save('trained_biases_b2.npy', b2)
    
    # Also save network parameters for reference
    network_params = {
        'hidden_layer_size': hidden_layer_size,
        'input_layer_size': input_layer_size,
        'output_layer_size': output_layer_size,
        'final_accuracy': per_correct[-1, 0]
    }
    np.save('network_params.npy', network_params)
    
    print("Weights and biases saved to:")
    print("- trained_weights_w1.npy")
    print("- trained_weights_w2.npy")
    print("- trained_biases_b1.npy")
    print("- trained_biases_b2.npy")
    print("- network_params.npy")
    
    # Create training progress plot
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, epochs + 1)
    
    # Plot accuracy over epochs (convert to percentage)
    plt.plot(epochs_range, per_correct.flatten() * 100, 'b-', linewidth=2, label=f'Hidden Layer Size = {hidden_layer_size}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Classification Accuracy (%)')
    plt.title(f'Neural Network Training Progress\n(Hidden Layer Size = {hidden_layer_size}, Learning Rate = {n})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, epochs)
    plt.ylim(0, 105)
    
    # Add final accuracy as text annotation
    plt.text(0.7 * epochs, 0.2 * 100, f'Final Accuracy: {per_correct[-1, 0]*100:.1f}%', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             fontsize=12, fontweight='bold')
    
    # Save the plot
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.savefig('training_progress.pdf', bbox_inches='tight')
    plt.show()

    print(f"\nTraining progress plot saved as 'training_progress.png' and 'training_progress.pdf'")
    
    return output, per_correct


if __name__ == "__main__":
    results = main()