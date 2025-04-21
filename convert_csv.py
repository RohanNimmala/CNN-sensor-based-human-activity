import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def convert_csv_to_numpy(csv_path, x_output_path, y_output_path, scaler_output_path=None):
    """
    Convert a CSV file to NumPy arrays for model training.
    The function assumes that the last column is the label and all other columns are features.
    
    Args:
        csv_path: Path to the CSV file
        x_output_path: Path to save the features array
        y_output_path: Path to save the labels array
        scaler_output_path: Optional path to save the scaler
    """
    # Load the CSV file
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract the label column (assumed to be the last column)
    print("Extracting features and labels...")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Reshape the data for HAR model (samples, channels, timesteps)
    # Assuming the data is structured as 9 channels with 128 timesteps each
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    # Try to determine a reasonable structure
    # For HAR data, common channel numbers are 3, 6, or 9 (accelerometer, gyroscope, magnetometer)
    # Common timesteps are powers of 2: 64, 128, 256, etc.
    if num_features % 9 == 0:
        num_channels = 9
    elif num_features % 6 == 0:
        num_channels = 6
    elif num_features % 3 == 0:
        num_channels = 3
    else:
        # Default to 9 channels and calculate timesteps
        num_channels = 9
    
    timesteps = num_features // num_channels
    print(f"Reshaping data with {num_samples} samples, {num_channels} channels, and {timesteps} timesteps...")
    
    # Reshape the data
    X_reshaped = X.reshape(num_samples, num_channels, timesteps)
    
    # Normalize the data
    if scaler_output_path:
        print("Normalizing data...")
        # Flatten for scaling
        X_flat = X.reshape(num_samples, -1)
        scaler = StandardScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        
        # Reshape back
        X_reshaped = X_flat_scaled.reshape(num_samples, num_channels, timesteps)
        
        # Save the scaler
        with open(scaler_output_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_output_path}")
    
    # Save the NumPy arrays
    print(f"Saving features to {x_output_path}...")
    np.save(x_output_path, X_reshaped)
    
    print(f"Saving labels to {y_output_path}...")
    np.save(y_output_path, y)
    
    print("Conversion complete!")
    return X_reshaped, y

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV file to NumPy arrays for HAR model')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file')
    parser.add_argument('--x_output', type=str, default='synthetic_har_data_X.npy', 
                        help='Path to save the features array')
    parser.add_argument('--y_output', type=str, default='synthetic_har_data_y.npy', 
                        help='Path to save the labels array')
    parser.add_argument('--scaler_output', type=str, default='har_scaler.pkl', 
                        help='Path to save the scaler')
    
    args = parser.parse_args()
    
    convert_csv_to_numpy(args.csv_path, args.x_output, args.y_output, args.scaler_output)