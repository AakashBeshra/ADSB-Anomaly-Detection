import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import pytz

class ADS_BDataProcessor:
    def __init__(self):
        self.last_update = None
        self.cache = {}
        
    async def fetch_live_data(self) -> List[Dict[str, Any]]:
        """Fetch real-time ADS-B data from OpenSky Network"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://opensky-network.org/api/states/all",
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_raw_data(data)
            except Exception as e:
                print(f"Error fetching data: {e}")
                return []
    
    def _process_raw_data(self, raw_data: Dict) -> List[Dict[str, Any]]:
        """Process raw ADS-B data into structured format"""
        processed_flights = []
        
        if 'states' not in raw_data:
            return processed_flights
            
        for state in raw_data['states']:
            if len(state) >= 17:
                flight_data = {
                    'icao24': state[0],
                    'callsign': state[1].strip() if state[1] else 'UNKNOWN',
                    'origin_country': state[2],
                    'time_position': state[3],
                    'last_contact': state[4],
                    'longitude': state[5],
                    'latitude': state[6],
                    'altitude': state[7] if state[7] else 0,
                    'velocity': state[9] if state[9] else 0,
                    'heading': state[10] if state[10] else 0,
                    'vertical_rate': state[11] if state[11] else 0,
                    'sensors': state[12],
                    'squawk': state[14],
                    'spi': state[15],
                    'position_source': state[16],
                    'category': state[17] if len(state) > 17 else 0,
                    'timestamp': datetime.now(pytz.UTC).isoformat()
                }
                processed_flights.append(flight_data)
        
        return processed_flights
    
    def extract_features(self, flight_data: Dict) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            flight_data.get('altitude', 0) or 0,
            flight_data.get('velocity', 0) or 0,
            flight_data.get('heading', 0) or 0,
            flight_data.get('vertical_rate', 0) or 0,
            flight_data.get('latitude', 0) or 0,
            flight_data.get('longitude', 0) or 0
        ]
        
        # Normalize features
        features = np.array(features).reshape(1, -1)
        return features
    
    def validate_flight_data(self, flight_data: Dict) -> bool:
        """Validate flight data for completeness"""
        required_fields = ['icao24', 'latitude', 'longitude', 'altitude']
        return all(field in flight_data and flight_data[field] is not None 
                  for field in required_fields)