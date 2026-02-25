# opensky_s3_downloader.py
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import numpy as np
import json
import gzip
import os
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Dict, Optional
import time

class OpenSkyS3Downloader:
    """Download and process OpenSky data from S3 bucket"""
    
    def __init__(self, output_dir: str = "data/opensky_processed"):
        # Setup S3 client for public bucket (no credentials needed)
        self.s3 = boto3.client('s3',
                              config=Config(signature_version=UNSIGNED),
                              region_name='us-east-1')
        
        self.bucket_name = 'opensky-network'
        self.states_prefix = 'data-samples/states/'
        self.output_dir = output_dir
        self.processed_dir = os.path.join(output_dir, "processed")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Airline to route mapping for realistic data
        self.airline_routes = {
            'UAL': [('JFK', 'LAX'), ('ORD', 'SFO'), ('EWR', 'SFO')],
            'AAL': [('DFW', 'ORD'), ('MIA', 'JFK'), ('LAX', 'JFK')],
            'DAL': [('ATL', 'LAX'), ('SLC', 'SEA'), ('DTW', 'MSP')],
            'BAW': [('LHR', 'JFK'), ('LHR', 'BOS'), ('MAN', 'JFK')],
            'AFR': [('CDG', 'JFK'), ('CDG', 'LAX'), ('CDG', 'MIA')],
            'DLH': [('FRA', 'JFK'), ('MUC', 'ORD'), ('FRA', 'LAX')],
            'KLM': [('AMS', 'JFK'), ('AMS', 'ATL'), ('AMS', 'IAH')],
            'SWA': [('DEN', 'LAS'), ('MDW', 'PHX'), ('BWI', 'TPA')],
            'JBU': [('JFK', 'FLL'), ('BOS', 'RSW'), ('FLL', 'BOS')],
            'VIR': [('LHR', 'MCO'), ('MAN', 'MCO'), ('LGW', 'LAS')],
            'ACA': [('YYZ', 'YVR'), ('YUL', 'YYC'), ('YYZ', 'YEG')],
            'ANA': [('HND', 'JFK'), ('NRT', 'LAX'), ('KIX', 'ORD')],
            'QFA': [('SYD', 'LAX'), ('MEL', 'DFW'), ('BNE', 'SFO')],
            'SIA': [('SIN', 'JFK'), ('SIN', 'LAX'), ('SIN', 'FRA')],
            'UAE': [('DXB', 'JFK'), ('DXB', 'LAX'), ('DXB', 'ORD')]
        }
        
    def list_available_files(self, limit: int = 100) -> List[str]:
        """List available state vector files in S3 bucket"""
        print("📋 Listing available files from OpenSky S3 bucket...")
        
        files = []
        continuation_token = None
        
        try:
            while True:
                # List objects with pagination
                list_kwargs = {
                    'Bucket': self.bucket_name,
                    'Prefix': self.states_prefix,
                    'MaxKeys': 1000
                }
                
                if continuation_token:
                    list_kwargs['ContinuationToken'] = continuation_token
                
                response = self.s3.list_objects_v2(**list_kwargs)
                
                # Filter for .json.gz files
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('.json.gz'):
                        # Extract date from filename
                        filename = obj['Key'].split('/')[-1]
                        if self._is_valid_filename(filename):
                            files.append(obj['Key'])
                
                # Check if more files are available
                if response.get('NextContinuationToken'):
                    continuation_token = response['NextContinuationToken']
                else:
                    break
                    
        except Exception as e:
            print(f"Error listing files: {str(e)}")
        
        print(f"Found {len(files)} valid state vector files")
        return files[:limit]
    
    def _is_valid_filename(self, filename: str) -> bool:
        """Check if filename matches expected pattern"""
        # Expected: YYYY-MM-DD-HH.json.gz
        parts = filename.replace('.json.gz', '').split('-')
        if len(parts) != 4:
            return False
        
        try:
            year, month, day, hour = map(int, parts)
            datetime(year, month, day, hour)
            return True
        except:
            return False
    
    def download_file(self, s3_key: str, output_filename: str) -> bool:
        """Download a single file from S3"""
        try:
            local_path = os.path.join(self.output_dir, output_filename)
            
            # Check if already downloaded
            if os.path.exists(local_path):
                print(f"  File already exists: {output_filename}")
                return True
            
            print(f"  Downloading {s3_key.split('/')[-1]}...")
            
            # Download with progress tracking
            start_time = time.time()
            self.s3.download_file(self.bucket_name, s3_key, local_path)
            
            # Get file size
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
            elapsed = time.time() - start_time
            
            print(f"    ✓ Saved as {output_filename} ({file_size:.1f} MB, {elapsed:.1f}s)")
            return True
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            return False
    
    def download_datasets(self, num_datasets: int = 30) -> List[str]:
        """Download specified number of datasets"""
        print(f"📥 Downloading {num_datasets} datasets from OpenSky...")
        
        # Get list of available files
        all_files = self.list_available_files(limit=200)
        
        if not all_files:
            print("No files found!")
            return []
        
        # Select diverse files (different dates/times)
        selected_files = self._select_diverse_files(all_files, num_datasets)
        
        print(f"Selected {len(selected_files)} files for download")
        
        # Download files
        downloaded_files = []
        
        for i, s3_key in enumerate(selected_files, 1):
            output_name = f"dataset_{i:02d}.json.gz"
            
            if self.download_file(s3_key, output_name):
                downloaded_files.append(output_name)
            
            # Small delay to be polite to the server
            time.sleep(0.1)
        
        print(f"\n✅ Downloaded {len(downloaded_files)}/{num_datasets} datasets")
        return downloaded_files
    
    def _select_diverse_files(self, files: List[str], count: int) -> List[str]:
        """Select diverse files (different dates, times)"""
        # Extract dates and group by date
        file_info = []
        for f in files:
            filename = f.split('/')[-1].replace('.json.gz', '')
            year, month, day, hour = map(int, filename.split('-'))
            file_info.append({
                'key': f,
                'date': f"{year}-{month:02d}-{day:02d}",
                'hour': hour,
                'datetime': datetime(year, month, day, hour)
            })
        
        # Sort by datetime
        file_info.sort(key=lambda x: x['datetime'])
        
        # Select diverse files
        selected = []
        
        # First, get one from each month if possible
        months_covered = set()
        for info in file_info:
            month_key = f"{info['datetime'].year}-{info['datetime'].month}"
            if month_key not in months_covered:
                selected.append(info['key'])
                months_covered.add(month_key)
                if len(selected) >= count:
                    break
        
        # If we need more, get different hours from different days
        if len(selected) < count:
            days_covered = set()
            for info in file_info:
                if info['key'] in selected:
                    continue
                
                day_key = info['date']
                if day_key not in days_covered:
                    selected.append(info['key'])
                    days_covered.add(day_key)
                    if len(selected) >= count:
                        break
        
        # If still need more, just take the earliest ones
        if len(selected) < count:
            for info in file_info:
                if info['key'] in selected:
                    continue
                selected.append(info['key'])
                if len(selected) >= count:
                    break
        
        return selected[:count]
    
    def process_json_to_csv(self, json_gz_path: str, output_csv_path: str) -> Optional[pd.DataFrame]:
        """Convert OpenSky JSON.gz to enriched CSV"""
        try:
            print(f"  Processing {os.path.basename(json_gz_path)}...")
            
            # Read gzipped JSON
            with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'states' not in data:
                print(f"    ✗ No 'states' found in file")
                return None
            
            timestamp = data.get('time', 0)
            dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
            
            processed_flights = []
            
            for state in data['states']:
                if len(state) >= 17 and state[5] and state[6]:  # Has lat/lon
                    flight = self._process_state_vector(state, timestamp)
                    
                    # Add timestamp info
                    flight['timestamp'] = dt.isoformat()
                    flight['hour_of_day'] = dt.hour
                    flight['day_of_week'] = dt.weekday()
                    
                    # Add route information
                    origin, destination = self._infer_route(flight['callsign'])
                    flight['origin'] = origin
                    flight['destination'] = destination
                    
                    # Calculate flight phase
                    flight['flight_phase'] = self._get_flight_phase(
                        flight['altitude_ft'], 
                        flight['velocity']
                    )
                    
                    # Add anomaly flags (for testing)
                    flight['is_anomaly'] = self._detect_anomaly(flight)
                    flight['anomaly_score'] = self._calculate_anomaly_score(flight)
                    
                    processed_flights.append(flight)
            
            if not processed_flights:
                print(f"    ✗ No valid flights found")
                return None
            
            df = pd.DataFrame(processed_flights)
            
            # Save to CSV
            df.to_csv(output_csv_path, index=False)
            
            print(f"    ✓ Processed {len(df)} flights")
            return df
            
        except Exception as e:
            print(f"    ✗ Processing error: {str(e)}")
            return None
    
    def _process_state_vector(self, state: List, timestamp: int) -> Dict:
        """Process a single state vector"""
        return {
            'icao24': state[0] or 'N/A',
            'callsign': (state[1] or '').strip(),
            'origin_country': state[2] or 'Unknown',
            'time_position': state[3] or 0,
            'last_contact': state[4] or 0,
            'longitude': float(state[5]) if state[5] else 0,
            'latitude': float(state[6]) if state[6] else 0,
            'baro_altitude': float(state[7]) if state[7] else 0,
            'on_ground': bool(state[8]) if state[8] is not None else False,
            'velocity': float(state[9]) if state[9] else 0,
            'true_track': float(state[10]) if state[10] else 0,
            'vertical_rate': float(state[11]) if state[11] else 0,
            'sensors': str(state[12]) if state[12] else '[]',
            'geo_altitude': float(state[13]) if state[13] else 0,
            'squawk': state[14] if state[14] else 'N/A',
            'spi': bool(state[15]) if state[15] is not None else False,
            'position_source': state[16] if state[16] else 0,
            'data_timestamp': timestamp,
            
            # Derived metrics
            'speed_knots': (float(state[9]) * 1.94384) if state[9] else 0,
            'altitude_ft': (float(state[7]) * 3.28084) if state[7] else 0,
            'geo_altitude_ft': (float(state[13]) * 3.28084) if state[13] else 0,
            'vertical_rate_fpm': (float(state[11]) * 196.85) if state[11] else 0,
            
            # Calculated fields
            'ground_speed': self._calculate_ground_speed(float(state[9]) if state[9] else 0),
            'is_military': self._is_military_callsign((state[1] or '').strip()),
            'airline': self._extract_airline((state[1] or '').strip())
        }
    
    def _infer_route(self, callsign: str) -> tuple:
        """Infer origin and destination based on callsign"""
        if not callsign or len(callsign) < 3:
            return ('UNK', 'UNK')
        
        # Extract airline code (first 3 letters)
        airline = callsign[:3]
        
        if airline in self.airline_routes:
            # Return a random route for this airline
            routes = self.airline_routes[airline]
            return routes[np.random.randint(0, len(routes))]
        
        # Default unknown
        return ('UNK', 'UNK')
    
    def _get_flight_phase(self, altitude_ft: float, speed_knots: float) -> str:
        """Determine flight phase based on altitude and speed"""
        if altitude_ft < 1000:
            return 'Takeoff/Landing'
        elif altitude_ft < 10000:
            return 'Climb'
        elif altitude_ft < 30000:
            return 'Cruise'
        elif altitude_ft >= 30000:
            return 'High Cruise'
        else:
            return 'Unknown'
    
    def _detect_anomaly(self, flight: Dict) -> bool:
        """Simple anomaly detection"""
        anomalies = []
        
        # Check altitude
        if flight['altitude_ft'] > 45000 or flight['altitude_ft'] < 500:
            anomalies.append('altitude')
        
        # Check speed
        if flight['speed_knots'] > 600 or flight['speed_knots'] < 100:
            anomalies.append('speed')
        
        # Check vertical rate
        if abs(flight['vertical_rate_fpm']) > 6000:
            anomalies.append('vertical_rate')
        
        # Check position (unusual locations)
        if abs(flight['latitude']) > 85 or abs(flight['longitude']) > 175:
            anomalies.append('position')
        
        # 10% chance of random anomaly for testing
        if np.random.random() < 0.1:
            anomalies.append('random')
        
        return len(anomalies) > 0
    
    def _calculate_anomaly_score(self, flight: Dict) -> float:
        """Calculate anomaly score (0-1)"""
        score = 0.0
        
        # Altitude anomaly (0-0.3)
        if flight['altitude_ft'] > 45000:
            score += 0.3
        elif flight['altitude_ft'] < 1000 and not flight['on_ground']:
            score += 0.2
        
        # Speed anomaly (0-0.3)
        if flight['speed_knots'] > 550:
            score += 0.3
        elif flight['speed_knots'] < 150 and flight['altitude_ft'] > 10000:
            score += 0.2
        
        # Vertical rate anomaly (0-0.2)
        if abs(flight['vertical_rate_fpm']) > 5000:
            score += 0.2
        
        # Position anomaly (0-0.2)
        if abs(flight['latitude']) > 80:
            score += 0.2
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_ground_speed(self, velocity_ms: float) -> float:
        """Convert m/s to knots"""
        return velocity_ms * 1.94384 if velocity_ms else 0
    
    def _is_military_callsign(self, callsign: str) -> bool:
        """Check if callsign appears to be military"""
        military_prefixes = ['RCH', 'SAM', 'SPAR', 'HKY', 'BOB', 'ED', 'ES']
        return any(callsign.startswith(prefix) for prefix in military_prefixes)
    
    def _extract_airline(self, callsign: str) -> str:
        """Extract airline code from callsign"""
        if len(callsign) >= 3:
            return callsign[:3]
        return 'UNK'
    
    def process_all_datasets(self, use_parallel: bool = True) -> Dict:
        """Process all downloaded datasets to CSV"""
        print("\n📊 Processing datasets to CSV format...")
        
        # Find all downloaded .json.gz files
        json_files = [f for f in os.listdir(self.output_dir) 
                     if f.endswith('.json.gz') and f.startswith('dataset_')]
        
        if not json_files:
            print("No datasets found to process!")
            return {}
        
        print(f"Found {len(json_files)} datasets to process")
        
        all_datasets_info = []
        
        if use_parallel:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for json_file in sorted(json_files):
                    json_path = os.path.join(self.output_dir, json_file)
                    csv_filename = json_file.replace('.json.gz', '.csv')
                    csv_path = os.path.join(self.processed_dir, csv_filename)
                    
                    futures.append(
                        executor.submit(self.process_json_to_csv, json_path, csv_path)
                    )
                
                # Collect results
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    df = future.result()
                    if df is not None:
                        dataset_id = f"dataset_{i:02d}"
                        all_datasets_info.append({
                            'id': dataset_id,
                            'num_flights': len(df),
                            'file_path': os.path.join(self.processed_dir, f"{dataset_id}.csv")
                        })
        else:
            # Sequential processing
            for i, json_file in enumerate(sorted(json_files), 1):
                json_path = os.path.join(self.output_dir, json_file)
                csv_filename = json_file.replace('.json.gz', '.csv')
                csv_path = os.path.join(self.processed_dir, csv_filename)
                
                df = self.process_json_to_csv(json_path, csv_path)
                
                if df is not None:
                    dataset_id = f"dataset_{i:02d}"
                    
                    # Rename to standardized name
                    final_path = os.path.join(self.processed_dir, f"{dataset_id}.csv")
                    df.to_csv(final_path, index=False)
                    
                    all_datasets_info.append({
                        'id': dataset_id,
                        'original_file': json_file,
                        'num_flights': len(df),
                        'file_path': final_path,
                        'timestamp': df['timestamp'].iloc[0] if not df.empty else ''
                    })
        
        # Create metadata file
        metadata = {
            'datasets': all_datasets_info,
            'total_datasets': len(all_datasets_info),
            'total_flights': sum(d['num_flights'] for d in all_datasets_info),
            'processing_date': datetime.now().isoformat(),
            'source': 'OpenSky Network S3 bucket',
            'columns_description': self._get_columns_description()
        }
        
        metadata_path = os.path.join(self.processed_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"   Processed datasets: {len(all_datasets_info)}")
        print(f"   Total flights: {metadata['total_flights']:,}")
        print(f"   Output directory: {self.processed_dir}/")
        print(f"   Metadata: {metadata_path}")
        
        return metadata
    
    def _get_columns_description(self) -> Dict:
        """Get description of all columns in processed CSV"""
        return {
            'icao24': 'Unique ICAO 24-bit aircraft address',
            'callsign': 'Flight callsign',
            'origin_country': 'Country of origin',
            'longitude': 'Longitude in decimal degrees',
            'latitude': 'Latitude in decimal degrees',
            'baro_altitude': 'Barometric altitude in meters',
            'altitude_ft': 'Altitude in feet',
            'velocity': 'Velocity in m/s',
            'speed_knots': 'Speed in knots',
            'true_track': 'True track in degrees',
            'vertical_rate': 'Vertical rate in m/s',
            'vertical_rate_fpm': 'Vertical rate in feet per minute',
            'on_ground': 'Whether aircraft is on ground',
            'origin': 'Inferred origin airport',
            'destination': 'Inferred destination airport',
            'flight_phase': 'Flight phase (Takeoff/Landing, Climb, Cruise, etc.)',
            'is_anomaly': 'Whether flight has anomalies',
            'anomaly_score': 'Anomaly score (0-1)',
            'timestamp': 'Timestamp of the data',
            'airline': 'Airline code extracted from callsign',
            'is_military': 'Whether flight appears to be military'
        }
    
    def create_sample_dashboard_data(self) -> None:
        """Create a sample dataset for the dashboard snapshot"""
        print("\n🎨 Creating sample dashboard data...")
        
        # Create a rich sample dataset
        sample_data = []
        
        # Create 50 sample flights with realistic data
        airlines = list(self.airline_routes.keys())
        
        for i in range(50):
            airline = np.random.choice(airlines)
            flight_num = f"{np.random.randint(100, 999)}"
            callsign = f"{airline}{flight_num}"
            
            origin, destination = self._infer_route(callsign)
            
            flight = {
                'flight_id': f"FL{i:03d}",
                'callsign': callsign,
                'airline': airline,
                'origin': origin,
                'destination': destination,
                'latitude': np.random.uniform(-90, 90),
                'longitude': np.random.uniform(-180, 180),
                'altitude_ft': np.random.choice([28000, 33000, 37000, 41000]),
                'speed_knots': np.random.uniform(420, 520),
                'heading': np.random.uniform(0, 360),
                'flight_phase': np.random.choice(['Climb', 'Cruise', 'Descent']),
                'status': np.random.choice(['Normal', 'Warning', 'Critical'], p=[0.85, 0.1, 0.05]),
                'passenger_count': np.random.randint(50, 300),
                'last_update': datetime.now().isoformat(),
                'estimated_arrival': (datetime.now() + timedelta(hours=np.random.randint(1, 6))).isoformat()
            }
            
            sample_data.append(flight)
        
        # Save sample data
        sample_path = os.path.join(self.processed_dir, 'dashboard_sample.json')
        with open(sample_path, 'w') as f:
            json.dump({'flights': sample_data}, f, indent=2)
        
        print(f"  ✓ Created dashboard sample: {sample_path}")
    
    def run_full_pipeline(self, num_datasets: int = 30) -> Dict:
        """Run complete pipeline: download -> process -> create samples"""
        print("=" * 60)
        print("🚀 OpenSky S3 Data Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Download datasets
        downloaded = self.download_datasets(num_datasets)
        
        if not downloaded:
            print("Failed to download datasets!")
            return {}
        
        # Step 2: Process to CSV
        metadata = self.process_all_datasets()
        
        # Step 3: Create sample dashboard data
        self.create_sample_dashboard_data()
        
        # Step 4: Verify first dataset
        if metadata['datasets']:
            first_dataset = metadata['datasets'][0]
            df = pd.read_csv(first_dataset['file_path'])
            print(f"\n📋 Sample from {first_dataset['id']}:")
            print(f"  Flights: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  First flight callsign: {df['callsign'].iloc[0] if not df.empty else 'N/A'}")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
        print("=" * 60)
        print("✅ Pipeline completed successfully!")
        
        return metadata


def main():
    """Main function to run the pipeline"""
    try:
        # Create downloader instance
        downloader = OpenSkyS3Downloader(output_dir="data/opensky_raw")
        
        # Run full pipeline
        metadata = downloader.run_full_pipeline(num_datasets=30)
        
        if metadata:
            print(f"\n📁 Your data is ready at: {downloader.processed_dir}/")
            print(f"📊 Total datasets: {metadata['total_datasets']}")
            print(f"✈️  Total flights: {metadata['total_flights']:,}")
            
            # Show available commands
            print("\n🔧 Next steps:")
            print("1. View first dataset: pandas.read_csv('data/opensky_raw/processed/dataset_01.csv')")
            print("2. Load metadata: json.load(open('data/opensky_raw/processed/metadata.json'))")
            print("3. Use in your dashboard: Update data_processor.py to read from these CSV files")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()