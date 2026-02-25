# backend/dataset_manager.py
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class DatasetManager:
    """Manager for handling 30 OpenSky datasets"""
    
    def __init__(self, data_dir: str = "data/opensky/processed"):
        self.data_dir = data_dir
        self.datasets = {}
        self.metadata = {}
        self.current_mode = "live"  # "live" or "snapshot"
        self.current_dataset = None
        self.current_time_index = 0
        
        # Load metadata
        self._load_metadata()
        # Load all datasets (lazy loading)
        self._load_all_datasets()
    
    def _load_metadata(self):
        """Load dataset metadata"""
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Create default metadata
            self.metadata = {
                'datasets': [
                    {'id': f'dataset_{i:02d}', 'flights': 100, 'description': f'Simulated dataset {i}'}
                    for i in range(1, 31)
                ],
                'total_datasets': 30,
                'total_flights': 3000
            }
    
    def _load_all_datasets(self):
        """Load all 30 datasets into memory (or prepare for lazy loading)"""
        print(f"📂 Loading 30 datasets from {self.data_dir}...")
        
        for i in range(1, 31):
            dataset_id = f"dataset_{i:02d}"
            file_path = os.path.join(self.data_dir, f"{dataset_id}.csv")
            
            if os.path.exists(file_path):
                # For large files, we might want lazy loading
                # For now, just track the file path
                self.datasets[dataset_id] = {
                    'path': file_path,
                    'loaded': False,
                    'data': None
                }
        
        print(f"✅ Found {len(self.datasets)} datasets")
    
    def get_dataset_list(self) -> List[Dict]:
        """Get list of all available datasets"""
        dataset_list = []
        for dataset_id, info in self.datasets.items():
            dataset_list.append({
                'id': dataset_id,
                'name': f"Dataset {dataset_id.split('_')[1]}",
                'path': info['path'],
                'flights': self._get_dataset_flight_count(dataset_id),
                'description': f"Flight data - {dataset_id}"
            })
        return dataset_list
    
    def _get_dataset_flight_count(self, dataset_id: str) -> int:
        """Get number of flights in a dataset"""
        if dataset_id in self.metadata.get('datasets', []):
            for ds in self.metadata['datasets']:
                if ds['id'] == dataset_id:
                    return ds.get('flights', 0)
        return 100  # Default
    
    def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load a specific dataset into memory"""
        if dataset_id not in self.datasets:
            print(f"Dataset {dataset_id} not found")
            return None
        
        info = self.datasets[dataset_id]
        
        if not info['loaded']:
            try:
                print(f"Loading {dataset_id}...")
                df = pd.read_csv(info['path'])
                info['data'] = df
                info['loaded'] = True
                info['size'] = len(df)
                print(f"  ✓ Loaded {len(df)} flights")
            except Exception as e:
                print(f"Error loading {dataset_id}: {str(e)}")
                return None
        
        return info['data']
    
    def get_snapshot(self, dataset_id: str, time_index: int = 0, num_flights: int = 50) -> List[Dict]:
        """Get snapshot from specific dataset at specific time"""
        df = self.load_dataset(dataset_id)
        if df is None:
            return []
        
        # If dataset has timestamp, filter by time
        if 'timestamp' in df.columns:
            # Get unique times and select one
            unique_times = sorted(df['timestamp'].unique())
            if time_index < len(unique_times):
                selected_time = unique_times[time_index]
                snapshot_df = df[df['timestamp'] == selected_time]
            else:
                snapshot_df = df.sample(min(num_flights, len(df)))
        else:
            # No timestamp, just get sample
            snapshot_df = df.sample(min(num_flights, len(df)))
        
        # Convert to list of dicts
        flights = snapshot_df.to_dict('records')
        
        # Add passenger data
        flights_with_passengers = self._add_passenger_data(flights)
        
        return flights_with_passengers
    
    def get_live_data(self) -> List[Dict]:
        """Generate live data by combining samples from all datasets"""
        print("Generating live flight data...")
        
        all_flights = []
        
        # Take samples from different datasets to simulate live data
        for dataset_id in list(self.datasets.keys())[:5]:  # Use first 5 datasets
            df = self.load_dataset(dataset_id)
            if df is not None and len(df) > 0:
                # Take random sample from each dataset
                sample_size = min(10, len(df))
                sample = df.sample(sample_size)
                
                # Update timestamps to current time
                for idx, row in sample.iterrows():
                    flight = row.to_dict()
                    flight['last_update'] = datetime.now().isoformat()
                    flight['source'] = 'live_simulation'
                    
                    # Simulate movement
                    flight = self._simulate_movement(flight)
                    
                    all_flights.append(flight)
        
        # Add passenger data
        flights_with_passengers = self._add_passenger_data(all_flights)
        
        return flights_with_passengers[:100]  # Limit to 100 flights
    
    def _add_passenger_data(self, flights: List[Dict]) -> List[Dict]:
        """Add synthetic passenger data to flights"""
        for flight in flights:
            # Generate passenger data for this flight
            passengers = self._generate_passengers_for_flight(flight.get('callsign', 'UNK'))
            flight['passengers'] = passengers[:5]  # Include first 5 passengers
            flight['passenger_count'] = len(passengers)
        
        return flights
    
    def _generate_passengers_for_flight(self, flight_callsign: str, num_passengers: int = None) -> List[Dict]:
        """Generate synthetic passenger data for a flight"""
        if num_passengers is None:
            num_passengers = np.random.randint(50, 300)
        
        passengers = []
        
        # Passenger names (simple list for demo)
        first_names = ['John', 'Jane', 'Robert', 'Maria', 'David', 'Sarah', 'Michael', 'Lisa', 
                      'James', 'Emily', 'William', 'Emma', 'Richard', 'Olivia', 'Charles', 'Sophia']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                     'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson']
        
        for i in range(num_passengers):
            passenger = {
                'passenger_id': f"PAX{np.random.randint(10000, 99999)}",
                'flight_callsign': flight_callsign,
                'name': f"{np.random.choice(first_names)} {np.random.choice(last_names)}",
                'seat': f"{np.random.randint(1, 40)}{np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}",
                'booking_class': np.random.choice(['Economy', 'Premium Economy', 'Business', 'First'], 
                                                 p=[0.6, 0.2, 0.15, 0.05]),
                'status': np.random.choice(['Checked-in', 'Boarded', 'Scheduled', 'No-show']),
                'check_in_time': (datetime.now() - timedelta(hours=np.random.randint(1, 6))).isoformat()
            }
            passengers.append(passenger)
        
        return passengers
    
    def _simulate_movement(self, flight: Dict) -> Dict:
        """Simulate flight movement for live updates"""
        # Add small random changes to position
        if 'longitude' in flight and 'latitude' in flight:
            flight['longitude'] += np.random.uniform(-0.1, 0.1)
            flight['latitude'] += np.random.uniform(-0.1, 0.1)
        
        # Update altitude slightly
        if 'altitude_ft' in flight:
            altitude_change = np.random.uniform(-100, 100)
            flight['altitude_ft'] = max(0, flight['altitude_ft'] + altitude_change)
        
        # Update speed slightly
        if 'speed_knots' in flight:
            speed_change = np.random.uniform(-5, 5)
            flight['speed_knots'] = max(0, flight['speed_knots'] + speed_change)
        
        return flight
    
    def set_mode(self, mode: str):
        """Set display mode: 'live' or 'snapshot'"""
        if mode in ['live', 'snapshot']:
            self.current_mode = mode
            print(f"Mode set to: {mode}")
        else:
            print(f"Invalid mode: {mode}")
    
    def get_current_data(self) -> List[Dict]:
        """Get data based on current mode"""
        if self.current_mode == 'live':
            return self.get_live_data()
        elif self.current_mode == 'snapshot' and self.current_dataset:
            return self.get_snapshot(self.current_dataset, self.current_time_index)
        else:
            return self.get_live_data()  # Default to live
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get detailed info about a dataset"""
        df = self.load_dataset(dataset_id)
        if df is None:
            return {}
        
        info = {
            'id': dataset_id,
            'total_flights': len(df),
            'columns': df.columns.tolist(),
            'airlines': df['airline'].unique().tolist() if 'airline' in df.columns else [],
            'anomaly_count': df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0,
            'time_range': self._get_time_range(df),
            'sample_flights': df.head(3).to_dict('records')
        }
        
        return info
    
    def _get_time_range(self, df: pd.DataFrame) -> Dict:
        """Get time range from dataset"""
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            return {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat()
            }
        return {'start': 'Unknown', 'end': 'Unknown'}