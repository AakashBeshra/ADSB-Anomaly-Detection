# process_opensky_data.py (in root folder)
import gzip
import json
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

def process_opensky_datasets():
    """Process all OpenSky JSON.gz files to CSV"""
    
    project_root = Path(__file__).parent
    
    # Define paths
    raw_dir = project_root / "data" / "opensky"
    processed_dir = project_root / "data" / "opensky_processed"
    
    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .json.gz files
    json_files = list(raw_dir.glob("*.json.gz"))
    
    if not json_files:
        print(f"❌ No JSON.gz files found in {raw_dir}")
        print("Run direct_download.py first!")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    all_datasets = []
    
    for i, json_path in enumerate(sorted(json_files), 1):
        print(f"\n[{i:02d}/{len(json_files)}] Processing {json_path.name}...")
        
        # Process file
        df = process_single_file(json_path)
        
        if df is not None and not df.empty:
            # Save as CSV
            csv_filename = f"dataset_{i:02d}.csv"
            csv_path = processed_dir / csv_filename
            df.to_csv(csv_path, index=False)
            
            all_datasets.append({
                'id': f"dataset_{i:02d}",
                'original_file': json_path.name,
                'csv_file': csv_filename,
                'num_flights': len(df),
                'file_size_mb': os.path.getsize(csv_path) / 1024 / 1024,
                'timestamp': df['timestamp'].iloc[0] if not df.empty else '',
                'path': str(csv_path.relative_to(project_root))
            })
            
            print(f"  ✓ Saved as data/opensky_processed/dataset_{i:02d}.csv")
            print(f"    Flights: {len(df):,} | Size: {all_datasets[-1]['file_size_mb']:.1f} MB")
    
    # Create metadata
    create_metadata(processed_dir, all_datasets, project_root)
    
    print(f"\n{'='*60}")
    print("✅ Processing Complete!")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"   Total datasets: {len(all_datasets)}")
    print(f"   Total flights: {sum(d['num_flights'] for d in all_datasets):,}")
    print(f"   Output location: data/opensky_processed/")
    print(f"\n🎯 Ready to use in your dashboard!")

def process_single_file(json_path):
    """Process a single OpenSky JSON.gz file"""
    try:
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'states' not in data:
            print(f"  ✗ No 'states' found in {json_path.name}")
            return None
        
        timestamp = data.get('time', 0)
        processed_flights = []
        
        for state in data['states']:
            if len(state) >= 17:
                flight = create_flight_dict(state, timestamp, json_path.name)
                processed_flights.append(flight)
        
        return pd.DataFrame(processed_flights)
        
    except Exception as e:
        print(f"  ✗ Error processing {json_path.name}: {str(e)}")
        return None

def create_flight_dict(state, timestamp, source_file):
    """Create flight dictionary from OpenSky state vector"""
    from datetime import datetime
    
    flight = {
        'icao24': state[0] or 'N/A',
        'callsign': (state[1] or '').strip() or 'N/A',
        'origin_country': state[2] or 'Unknown',
        'time_position': state[3] or 0,
        'last_contact': state[4] or 0,
        'longitude': state[5] or 0,
        'latitude': state[6] or 0,
        'baro_altitude': state[7] or 0,
        'on_ground': bool(state[8]) if state[8] is not None else False,
        'velocity': state[9] or 0,
        'true_track': state[10] or 0,
        'vertical_rate': state[11] or 0,
        'sensors': str(state[12]) if state[12] else '[]',
        'geo_altitude': state[13] or 0,
        'squawk': state[14] or 'N/A',
        'spi': bool(state[15]) if state[15] is not None else False,
        'position_source': state[16] or 0,
        'timestamp': datetime.fromtimestamp(timestamp).isoformat() if timestamp else '',
        'dataset_timestamp': timestamp,
        'file_source': source_file,
        'speed_knots': (state[9] or 0) * 1.94384,
        'altitude_ft': (state[7] or 0) * 3.28084,
        'origin': 'Unknown',
        'destination': 'Unknown'
    }
    
    # Add route based on callsign pattern
    flight['origin'], flight['destination'] = infer_route(flight['callsign'])
    
    return flight

def infer_route(callsign):
    """Simple route inference"""
    common_routes = {
        'UAL': ('JFK', 'LAX'),
        'AAL': ('DFW', 'ORD'),
        'DAL': ('ATL', 'JFK'),
        'BAW': ('LHR', 'JFK'),
        'AFR': ('CDG', 'JFK'),
        'DLH': ('FRA', 'JFK'),
        'KLM': ('AMS', 'JFK'),
        'SWA': ('DEN', 'LAS'),
        'JBU': ('JFK', 'BOS'),
        'VIR': ('LHR', 'MCO'),
    }
    
    if callsign != 'N/A':
        for airline, route in common_routes.items():
            if callsign.startswith(airline):
                return route
    
    return ('Unknown', 'Unknown')

def create_metadata(processed_dir, datasets, project_root):
    """Create metadata.json file"""
    
    metadata = {
        'project': 'ADS-B Anomaly Detection System',
        'datasets': datasets,
        'summary': {
            'total_datasets': len(datasets),
            'total_flights': sum(d['num_flights'] for d in datasets),
            'average_flights_per_dataset': sum(d['num_flights'] for d in datasets) // len(datasets) if datasets else 0,
            'processing_date': datetime.now().isoformat(),
            'source': 'OpenSky Network S3 bucket',
            'data_structure': 'State vectors (position, altitude, velocity, etc.)'
        },
        'columns_description': {
            'icao24': 'Unique aircraft identifier',
            'callsign': 'Flight number (e.g., UAL123)',
            'origin_country': 'Aircraft registration country',
            'time_position': 'Unix timestamp of position report',
            'last_contact': 'Unix timestamp of last contact',
            'longitude': 'Longitude in decimal degrees',
            'latitude': 'Latitude in decimal degrees',
            'baro_altitude': 'Altitude in meters (barometric)',
            'on_ground': 'Whether aircraft is on ground',
            'velocity': 'Ground speed in m/s',
            'true_track': 'Direction in decimal degrees',
            'vertical_rate': 'Climb/descent rate in m/s',
            'sensors': 'Sensor IDs that contributed',
            'geo_altitude': 'Altitude in meters (geometric)',
            'squawk': 'Transponder code',
            'spi': 'Special purpose indicator',
            'position_source': 'Source of position data',
            'timestamp': 'Human-readable timestamp',
            'dataset_timestamp': 'Original dataset timestamp',
            'file_source': 'Source filename',
            'speed_knots': 'Ground speed in knots',
            'altitude_ft': 'Altitude in feet',
            'origin': 'Inferred departure airport',
            'destination': 'Inferred arrival airport'
        },
        'paths': {
            'project_root': str(project_root),
            'raw_data': 'data/opensky/',
            'processed_data': 'data/opensky_processed/',
            'passenger_data': 'data/passenger/',
            'frontend': 'frontend/',
            'backend': 'backend/'
        }
    }
    
    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: data/opensky_processed/metadata.json")

if __name__ == "__main__":
    process_opensky_datasets()