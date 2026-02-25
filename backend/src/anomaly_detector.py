from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
import pytz
from collections import deque
import asyncio

class AnomalyDetector:
    def __init__(self, ml_model):
        self.ml_model = ml_model
        self.flight_history = {}
        self.max_history = 100
        self.anomaly_types = {
            'sudden_altitude_change': 'Sudden Altitude Change',
            'unusual_speed': 'Unusual Speed Pattern',
            'erratic_trajectory': 'Erratic Flight Path',
            'stale_position': 'Stale Position Data',
            'airspace_violation': 'Airspace Violation'
        }
        
    async def detect_anomalies(self, flight_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in flight data"""
        anomalies = []
        
        for flight in flight_data:
            # Extract features for ML
            features = self._extract_ml_features(flight)
            
            # Rule-based anomaly detection
            rule_anomalies = self._rule_based_detection(flight)
            
            # ML-based anomaly detection
            if self.ml_model.is_trained:
                ml_score, ml_anomaly = self.ml_model.predict_anomaly(features)
                flight['anomaly_score'] = float(ml_score[0])
                flight['is_anomaly_ml'] = bool(ml_anomaly[0])
                
                if ml_anomaly[0]:
                    anomalies.append({
                        'flight_id': flight['icao24'],
                        'callsign': flight.get('callsign', 'UNKNOWN'),
                        'anomaly_type': 'ML Detected Anomaly',
                        'severity': 'high' if ml_score[0] > 0.9 else 'medium',
                        'score': float(ml_score[0]),
                        'timestamp': flight.get('timestamp'),
                        'details': {
                            'altitude': flight.get('altitude'),
                            'velocity': flight.get('velocity'),
                            'location': {
                                'lat': flight.get('latitude'),
                                'lng': flight.get('longitude')
                            }
                        }
                    })
            
            # Combine with rule-based anomalies
            anomalies.extend(rule_anomalies)
            
            # Update flight history
            self._update_flight_history(flight)
        
        return anomalies
    
    def _extract_ml_features(self, flight: Dict[str, Any]) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            flight.get('altitude', 0) or 0,
            flight.get('velocity', 0) or 0,
            flight.get('vertical_rate', 0) or 0,
            flight.get('heading', 0) or 0,
            abs(flight.get('latitude', 0) or 0),
            abs(flight.get('longitude', 0) or 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def _rule_based_detection(self, flight: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rule-based anomaly detection"""
        anomalies = []
        
        # Check for sudden altitude changes
        if flight.get('vertical_rate', 0):
            if abs(flight['vertical_rate']) > 3000:  # ft/min
                anomalies.append({
                    'flight_id': flight['icao24'],
                    'callsign': flight.get('callsign', 'UNKNOWN'),
                    'anomaly_type': 'sudden_altitude_change',
                    'severity': 'high',
                    'score': 0.95,
                    'timestamp': flight.get('timestamp'),
                    'details': {
                        'vertical_rate': flight['vertical_rate'],
                        'current_altitude': flight.get('altitude')
                    }
                })
        
        # Check for unusual speed
        velocity = flight.get('velocity', 0)
        if velocity > 600 or (velocity < 50 and flight.get('altitude', 0) > 10000):
            anomalies.append({
                'flight_id': flight['icao24'],
                'callsign': flight.get('callsign', 'UNKNOWN'),
                'anomaly_type': 'unusual_speed',
                'severity': 'medium',
                'score': 0.85,
                'timestamp': flight.get('timestamp'),
                'details': {
                    'velocity': velocity,
                    'altitude': flight.get('altitude')
                }
            })
        
        # Check for stale data
        if 'time_position' in flight and flight['time_position']:
            position_age = datetime.now(pytz.UTC).timestamp() - flight['time_position']
            if position_age > 300:  # 5 minutes
                anomalies.append({
                    'flight_id': flight['icao24'],
                    'callsign': flight.get('callsign', 'UNKNOWN'),
                    'anomaly_type': 'stale_position',
                    'severity': 'low',
                    'score': 0.7,
                    'timestamp': flight.get('timestamp'),
                    'details': {
                        'position_age_seconds': position_age
                    }
                })
        
        return anomalies
    
    def _update_flight_history(self, flight: Dict[str, Any]):
        """Update flight history for trend analysis"""
        flight_id = flight['icao24']
        
        if flight_id not in self.flight_history:
            self.flight_history[flight_id] = deque(maxlen=self.max_history)
        
        self.flight_history[flight_id].append({
            'timestamp': datetime.now(pytz.UTC),
            'position': (flight.get('latitude'), flight.get('longitude')),
            'altitude': flight.get('altitude'),
            'velocity': flight.get('velocity'),
            'heading': flight.get('heading')
        })
