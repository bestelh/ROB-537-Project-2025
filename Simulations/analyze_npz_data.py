"""
Script to analyze the .npz simulation results file.
This script reads the .npz file and explores its structure and contents.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def analyze_npz_file(file_path):
    """
    Load and analyze the contents of an .npz file.
    
    Parameters:
    file_path (str): Path to the .npz file
    
    Returns:
    dict: Dictionary containing the loaded data and analysis results
    """
    
    print(f"Analyzing file: {file_path}")
    print("=" * 50)
    
    # Load the .npz file
    try:
        npz_data = np.load(file_path, allow_pickle=True)
        print("✓ Successfully loaded .npz file")
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None
    
    # Show the top-level keys in the .npz file
    print(f"\nTop-level keys in .npz file: {list(npz_data.keys())}")
    
    analysis_results = {}
    
    # Analyze each key in the .npz file
    for key in npz_data.keys():
        print(f"\n--- Analyzing key: '{key}' ---")
        data = npz_data[key]
        
        print(f"Type: {type(data)}")
        print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"Data type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        
        # If it's an array of objects (likely dictionaries from our simulation)
        if data.dtype == object:
            print(f"Number of simulation results: {len(data)}")
            
            if len(data) > 0:
                # Examine the first result to understand structure
                first_result = data[0]
                if isinstance(first_result, dict):
                    print(f"\nStructure of individual simulation results:")
                    print(f"Keys in each result: {list(first_result.keys())}")
                    
                    # Analyze each field in the simulation results
                    print(f"\nField analysis:")
                    for field_key in first_result.keys():
                        field_data = first_result[field_key]
                        field_type = type(field_data)
                        field_shape = field_data.shape if hasattr(field_data, 'shape') else 'scalar'
                        print(f"  {field_key}: {field_type}, shape={field_shape}")
                    
                    # Create a summary of all simulation parameters and results
                    analysis_results['simulation_data'] = data
                    analysis_results['num_simulations'] = len(data)
                    analysis_results['field_names'] = list(first_result.keys())
                    
                    # Extract key statistics
                    print(f"\n--- Data Statistics ---")
                    
                    # Collect all scalar fields for statistical analysis
                    scalar_fields = {}
                    for field_key in first_result.keys():
                        if not hasattr(first_result[field_key], 'shape') or \
                           (hasattr(first_result[field_key], 'shape') and len(first_result[field_key].shape) == 0):
                            # This is a scalar field
                            values = [result[field_key] for result in data if field_key in result]
                            if len(values) > 0 and isinstance(values[0], (int, float)):
                                scalar_fields[field_key] = values
                    
                    # Display statistics for scalar fields
                    if scalar_fields:
                        stats_df = pd.DataFrame()
                        for field, values in scalar_fields.items():
                            stats_df[field] = [
                                np.mean(values),
                                np.std(values),
                                np.min(values),
                                np.max(values),
                                len(values)
                            ]
                        
                        stats_df.index = ['Mean', 'Std', 'Min', 'Max', 'Count']
                        print(f"\nStatistics for scalar fields:")
                        print(stats_df)
                        
                        analysis_results['statistics'] = stats_df
    
    # Close the npz file
    npz_data.close()
    
    return analysis_results

def plot_sample_data(analysis_results, num_samples=5):
    """
    Plot sample force distributions and other data from the simulation results.
    
    Parameters:
    analysis_results (dict): Results from analyze_npz_file function
    num_samples (int): Number of random samples to plot
    """
    
    if 'simulation_data' not in analysis_results:
        print("No simulation data found for plotting.")
        return
    
    data = analysis_results['simulation_data']
    
    # Plot sample force distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Sample Force Distributions and Results', fontsize=14)
    
    # Select random samples
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    
    # Plot 1: Force distributions in X, Y, Z
    ax1 = axes[0, 0]
    for i, idx in enumerate(indices):
        result = data[idx]
        if 's_force' in result and 'f_dist_fx' in result:
            s = result['s_force']
            fx = result['f_dist_fx']
            fy = result['f_dist_fy']
            fz = result['f_dist_fz']
            
            ax1.plot(s, fx, label=f'Sample {i+1} - Fx', alpha=0.7)
            
    ax1.set_xlabel('Arc Length s')
    ax1.set_ylabel('Force (N)')
    ax1.set_title('Sample Force Distributions (X-direction)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Input vs Deformed Lengths (L1)
    ax2 = axes[0, 1]
    l1_inputs = []
    l1_defs = []
    
    for result in data:
        for seg in range(1, 5):  # Segments 1-4
            if f'L1_input_{seg}' in result and f'L1_def_{seg}' in result:
                l1_inputs.append(result[f'L1_input_{seg}'])
                l1_defs.append(result[f'L1_def_{seg}'])
    
    if l1_inputs and l1_defs:
        ax2.scatter(l1_inputs, l1_defs, alpha=0.6, s=10)
        # Plot y=x line for reference
        min_val = min(min(l1_inputs), min(l1_defs))
        max_val = max(max(l1_inputs), max(l1_defs))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        ax2.set_xlabel('Input Length L1')
        ax2.set_ylabel('Deformed Length L1')
        ax2.set_title('Input vs Deformed Actuator Lengths (L1)')
        ax2.grid(True)
        ax2.legend()
    
    # Plot 3: Force amplitude distribution
    ax3 = axes[1, 0]
    force_amps_x = [result['force_amp_x'] for result in data if 'force_amp_x' in result]
    force_amps_y = [result['force_amp_y'] for result in data if 'force_amp_y' in result]
    force_amps_z = [result['force_amp_z'] for result in data if 'force_amp_z' in result]
    
    ax3.hist(force_amps_x, bins=30, alpha=0.7, label='Fx amplitude')
    ax3.hist(force_amps_y, bins=30, alpha=0.7, label='Fy amplitude')
    ax3.hist(force_amps_z, bins=30, alpha=0.7, label='Fz amplitude')
    ax3.set_xlabel('Force Amplitude (N)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Force Amplitudes')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Method usage
    ax4 = axes[1, 1]
    methods_x = [result['method_x'] for result in data if 'method_x' in result]
    method_counts = pd.Series(methods_x).value_counts()
    
    ax4.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    ax4.set_title('Distribution of Force Generation Methods (X-direction)')
    
    plt.tight_layout()
    plt.show()

def extract_data_labels(analysis_results):
    """
    Extract and display all data labels/field names in the simulation results.
    
    Parameters:
    analysis_results (dict): Results from analyze_npz_file function
    
    Returns:
    list: List of all field names
    """
    
    if 'field_names' not in analysis_results:
        print("No field names found in analysis results.")
        return []
    
    field_names = analysis_results['field_names']
    
    print("\n" + "="*60)
    print("DATA LABELS / FIELD NAMES IN SIMULATION RESULTS")
    print("="*60)
    
    # Categorize the fields
    input_fields = []
    output_fields = []
    force_fields = []
    method_fields = []
    other_fields = []
    
    for field in field_names:
        if 'input' in field.lower():
            input_fields.append(field)
        elif 'def' in field.lower():
            output_fields.append(field)
        elif 'force' in field.lower() or 'f_dist' in field.lower():
            force_fields.append(field)
        elif 'method' in field.lower():
            method_fields.append(field)
        else:
            other_fields.append(field)
    
    print(f"\n📥 INPUT PARAMETERS ({len(input_fields)} fields):")
    for field in sorted(input_fields):
        print(f"  • {field}")
    
    print(f"\n📤 OUTPUT RESULTS ({len(output_fields)} fields):")
    for field in sorted(output_fields):
        print(f"  • {field}")
    
    print(f"\n🔧 FORCE PARAMETERS ({len(force_fields)} fields):")
    for field in sorted(force_fields):
        print(f"  • {field}")
    
    print(f"\n⚙️  METHOD PARAMETERS ({len(method_fields)} fields):")
    for field in sorted(method_fields):
        print(f"  • {field}")
    
    if other_fields:
        print(f"\n🔍 OTHER FIELDS ({len(other_fields)} fields):")
        for field in sorted(other_fields):
            print(f"  • {field}")
    
    print(f"\n📊 TOTAL: {len(field_names)} data fields per simulation")
    
    return field_names

def main():
    """Main function to run the analysis."""
    
    # File path to the .npz file - try multiple possible names
    possible_files = [
        "simulation_results_partial.npz"
    ]
    
    npz_file_path = None
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent
    
    # Check current directory and script directory
    for directory in [current_dir, script_dir]:
        for filename in possible_files:
            filepath = directory / filename
            if filepath.exists():
                npz_file_path = str(filepath)
                break
        if npz_file_path:
            break
    
    if npz_file_path is None:
        print("No simulation results file found!")
        print(f"Current directory: {current_dir}")
        print(f"Script directory: {script_dir}")
        print(f"Looking for files: {possible_files}")
        print("\nFiles in current directory:")
        try:
            for file in current_dir.glob("*.npz"):
                print(f"  - {file.name}")
        except:
            print("  Could not list files")
        return
    
    print(f"Found file: {npz_file_path}")
    
    # Analyze the .npz file
    results = analyze_npz_file(npz_file_path)
    
    
    if results is None:
        return
    
    # Extract and display data labels
    field_names = extract_data_labels(results)
    
    # Plot sample data
    print(f"\n📈 Generating sample plots...")
    plot_sample_data(results, num_samples=3)
    
    # Provide usage examples
    print(f"\n" + "="*60)
    print("HOW TO ACCESS THE DATA IN YOUR OWN SCRIPTS")
    print("="*60)
    print(f"""
# Load the data:
import numpy as np
data = np.load('{npz_file_path}', allow_pickle=True)
simulation_results = data['data']  # Array of simulation dictionaries

# Access individual simulation results:
first_sim = simulation_results[0]
print("Available fields:", list(first_sim.keys()))

# Extract specific data:
force_amplitudes_x = [sim['force_amp_x'] for sim in simulation_results]
input_lengths_L1_seg1 = [sim['L1_input_1'] for sim in simulation_results]
deformed_lengths_L1_seg1 = [sim['L1_def_1'] for sim in simulation_results]

# Access force distributions:
s_positions = first_sim['s_force']      # Arc length positions
fx_profile = first_sim['f_dist_fx']     # Force distribution in X
fy_profile = first_sim['f_dist_fy']     # Force distribution in Y  
fz_profile = first_sim['f_dist_fz']     # Force distribution in Z

# Don't forget to close the file when done:
data.close()
""")

if __name__ == "__main__":
    main()