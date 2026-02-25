import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data
from isolation_forest import train_isolation_forest
from lstm_autoencoder import train_lstm_autoencoder
import joblib
import os

def main():
    # Create models directory
    os.makedirs('../backend/models', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_normalized = load_and_preprocess_data()
    
    if X_normalized is None or len(X_normalized) == 0:
        print("No data available for training. Using synthetic data...")
        # Generate synthetic flight data for training
        np.random.seed(42)
        n_samples = 1000
        X_normalized = np.random.randn(n_samples, 6)
        X_normalized[:, 0] = np.random.uniform(0, 40000, n_samples)  # altitude
        X_normalized[:, 1] = np.random.uniform(0, 600, n_samples)    # velocity
        X_normalized[:, 2] = np.random.uniform(-3000, 3000, n_samples)  # vertical rate
        X_normalized[:, 3] = np.random.uniform(0, 360, n_samples)    # heading
        X_normalized[:, 4] = np.random.uniform(-90, 90, n_samples)   # latitude
        X_normalized[:, 5] = np.random.uniform(-180, 180, n_samples) # longitude
    
    print(f"Training data shape: {X_normalized.shape}")
    
    # Train Isolation Forest
    print("\nTraining Isolation Forest...")
    iso_forest = train_isolation_forest(X_normalized, contamination=0.1)
    
    # Train LSTM Autoencoder
    print("\nTraining LSTM Autoencoder...")
    lstm_ae = train_lstm_autoencoder(X_normalized, epochs=50)
    
    # Save models
    print("\nSaving models...")
    joblib.dump(iso_forest, '../backend/models/isolation_forest.joblib')
    lstm_ae.save('../backend/models/lstm_autoencoder.h5')
    
    # Create and save scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_normalized)
    joblib.dump(scaler, '../backend/models/scaler.joblib')
    
    print("Training completed successfully!")
    print("Models saved in backend/models/ directory")

if __name__ == "__main__":
    main()