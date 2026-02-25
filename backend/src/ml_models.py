import numpy as np
from typing import Tuple, List, Optional
import joblib
import warnings
import sys

# Conditional imports based on availability
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model, save_model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

class FallbackAnomalyDetector:
    """Fallback anomaly detector when ML libraries aren't available"""
    def __init__(self):
        self.threshold = 0.85
        
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple rule-based fallback"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Simple heuristic: high altitude + low speed = anomaly
        scores = np.zeros(len(X))
        for i, features in enumerate(X):
            if len(features) >= 2:
                altitude = abs(features[0])
                speed = abs(features[1])
                
                # Simple heuristic score
                if altitude > 30000 and speed < 100:
                    scores[i] = 0.9
                elif altitude < 1000 and speed > 500:
                    scores[i] = 0.8
                else:
                    scores[i] = 0.1
        
        predictions = (scores > self.threshold).astype(int)
        return scores, predictions

class AnomalyDetectionModels:
    def __init__(self):
        self.isolation_forest = None
        self.lstm_autoencoder = None
        self.one_class_svm = None
        self.scaler = None
        self.fallback_detector = FallbackAnomalyDetector()
        self.is_trained = False
        
        # Check library availability
        self.sklearn_available = SKLEARN_AVAILABLE
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        
        if self.sklearn_available:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        
        print(f"🤖 ML Libraries: scikit-learn={self.sklearn_available}, TensorFlow={self.tensorflow_available}")
    
    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.1):
        """Train Isolation Forest model"""
        if not self.sklearn_available:
            print("⚠️ scikit-learn not available for training Isolation Forest")
            return
            
        try:
            from sklearn.ensemble import IsolationForest
            self.isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            X_scaled = self.scaler.fit_transform(X)
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
        except Exception as e:
            print(f"❌ Error training Isolation Forest: {e}")
    
    def build_lstm_autoencoder(self, input_shape: Tuple[int, int]):
        """Build LSTM Autoencoder for sequential anomaly detection"""
        if not self.tensorflow_available:
            print("⚠️ TensorFlow not available for LSTM Autoencoder")
            return None
            
        try:
            # Encoder
            inputs = Input(shape=input_shape)
            encoded = LSTM(32, activation='relu', return_sequences=True)(inputs)
            encoded = LSTM(16, activation='relu', return_sequences=False)(encoded)
            
            # Decoder
            decoded = RepeatVector(input_shape[0])(encoded)
            decoded = LSTM(16, activation='relu', return_sequences=True)(decoded)
            decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
            decoded = TimeDistributed(Dense(input_shape[1]))(decoded)
            
            # Autoencoder
            autoencoder = Model(inputs, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            return autoencoder
        except Exception as e:
            print(f"❌ Error building LSTM Autoencoder: {e}")
            return None
    
    def train_lstm_autoencoder(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train LSTM Autoencoder"""
        if not self.tensorflow_available or not self.scaler:
            print("⚠️ TensorFlow or scaler not available for LSTM training")
            return
            
        try:
            # Scale the data
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape for LSTM [samples, timesteps, features]
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            # Build and train model
            self.lstm_autoencoder = self.build_lstm_autoencoder((1, X.shape[1]))
            
            if self.lstm_autoencoder:
                early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                
                self.lstm_autoencoder.fit(
                    X_reshaped, X_reshaped,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
                self.is_trained = True
        except Exception as e:
            print(f"❌ Error training LSTM Autoencoder: {e}")
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using available models"""
        if not self.is_trained:
            # Use fallback detector
            return self.fallback_detector.predict_anomaly(X)
        
        try:
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Scale the data if scaler is available
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Use available models
            model_scores = []
            
            # Isolation Forest
            if self.isolation_forest:
                if_scores = -self.isolation_forest.score_samples(X_scaled)
                if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
                model_scores.append(if_scores_norm)
            
            # LSTM Autoencoder
            if self.lstm_autoencoder and self.tensorflow_available:
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                try:
                    reconstructions = self.lstm_autoencoder.predict(X_reshaped, verbose=0)
                    mse = np.mean(np.square(X_reshaped - reconstructions), axis=(1, 2))
                    mse_norm = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
                    model_scores.append(mse_norm)
                except Exception as e:
                    print(f"⚠️ LSTM prediction error: {e}")
            
            # Combine scores from available models
            if model_scores:
                ensemble_scores = np.mean(model_scores, axis=0)
            else:
                # Use fallback if no models available
                return self.fallback_detector.predict_anomaly(X)
            
            # Normalize final scores
            normalized_scores = (ensemble_scores - ensemble_scores.min()) / (ensemble_scores.max() - ensemble_scores.min() + 1e-8)
            
            # Predict anomalies (threshold = 0.85)
            predictions = (normalized_scores > 0.85).astype(int)
            
            return normalized_scores, predictions
            
        except Exception as e:
            print(f"❌ Error in anomaly prediction: {e}")
            # Fallback to rule-based detection
            return self.fallback_detector.predict_anomaly(X)
    
    def save_models(self, path: str):
        """Save trained models"""
        import os
        
        os.makedirs(path, exist_ok=True)
        
        try:
            if self.isolation_forest:
                joblib.dump(self.isolation_forest, f"{path}/isolation_forest.joblib")
            
            if self.lstm_autoencoder and self.tensorflow_available:
                # Save TensorFlow model
                self.lstm_autoencoder.save(f"{path}/lstm_autoencoder.keras")
            
            if self.scaler:
                joblib.dump(self.scaler, f"{path}/scaler.joblib")
            
            print(f"✅ Models saved to {path}")
        except Exception as e:
            print(f"❌ Error saving models: {e}")
        
    def load_models(self, path: str):
        """Load trained models"""
        import os
        
        try:
            # Load scaler
            scaler_path = f"{path}/scaler.joblib"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler loaded")
            else:
                # Create new scaler if none exists
                if self.sklearn_available:
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
            
            # Load Isolation Forest
            iso_path = f"{path}/isolation_forest.joblib"
            if os.path.exists(iso_path):
                self.isolation_forest = joblib.load(iso_path)
                print("✅ Isolation Forest loaded")
            
            # Load LSTM Autoencoder
            lstm_path = f"{path}/lstm_autoencoder.keras"
            if os.path.exists(lstm_path) and self.tensorflow_available:
                self.lstm_autoencoder = load_model(lstm_path)
                print("✅ LSTM Autoencoder loaded")
            
            # Mark as trained if at least one model loaded
            if self.isolation_forest or self.lstm_autoencoder:
                self.is_trained = True
                print("✅ ML models loaded successfully")
            else:
                print("⚠️ No ML models found, using rule-based detection")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.is_trained = False