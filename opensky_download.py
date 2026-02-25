# opensky_download_fixed.py
import requests
import os
import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
import time

def download_opensky_data():
    """Download OpenSky data using direct HTTP requests"""
    
    print("📡 Downloading OpenSky datasets...")
    
    # Create directories
    os.makedirs("data/opensky/raw", exist_ok=True)
    os.makedirs("data/opensky/processed", exist_ok=True)
    
    # OpenSky data URLs - using direct HTTP access
    data_urls = [
        # 2023 data - different hours
        "https://opensky-network.org/datasets/states/2023-01-01-00.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-01.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-02.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-03.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-04.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-05.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-06.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-07.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-08.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-09.json.gz",
        
        # Different hours for variety
        "https://opensky-network.org/datasets/states/2023-01-01-10.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-11.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-12.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-13.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-14.json.gz",
        
        # Different days
        "https://opensky-network.org/datasets/states/2023-01-02-10.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-03-14.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-04-16.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-05-20.json.gz",
        
        # Different months
        "https://opensky-network.org/datasets/states/2023-03-15-08.json.gz",
        "https://opensky-network.org/datasets/states/2023-06-15-12.json.gz",
        "https://opensky-network.org/datasets/states/2023-09-15-16.json.gz",
        "https://opensky-network.org/datasets/states/2023-12-15-20.json.gz",
        
        # 2022 data
        "https://opensky-network.org/datasets/states/2022-07-01-06.json.gz",
        "https://opensky-network.org/datasets/states/2022-10-01-12.json.gz",
        
        # 2021 data
        "https://opensky-network.org/datasets/states/2021-04-01-18.json.gz",
        "https://opensky-network.org/datasets/states/2021-08-01-00.json.gz",
        
        # More varied times
        "https://opensky-network.org/datasets/states/2023-01-01-15.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-18.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-21.json.gz",
        "https://opensky-network.org/datasets/states/2023-01-01-23.json.gz",
    ]
    
    print(f"Downloading {len(data_urls)} datasets...")
    
    successful_downloads = []
    
    for i, url in enumerate(data_urls, 1):
        try:
            filename = url.split('/')[-1]
            dataset_name = f"dataset_{i:02d}.json.gz"
            raw_path = os.path.join("data/opensky/raw", dataset_name)
            
            # Check if already downloaded
            if os.path.exists(raw_path):
                print(f"[{i:02d}/{len(data_urls)}] Already exists: {dataset_name}")
                successful_downloads.append((dataset_name, url))
                continue
            
            print(f"[{i:02d}/{len(data_urls)}] Downloading {filename}...")
            
            # Download with timeout and retry
            success = False
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        # Save the file
                        with open(raw_path, 'wb') as f:
                            f.write(response.content)
                        
                        file_size = os.path.getsize(raw_path) / (1024 * 1024)
                        print(f"    ✓ Saved as {dataset_name} ({file_size:.1f} MB)")
                        successful_downloads.append((dataset_name, url))
                        success = True
                        break
                    else:
                        print(f"    ✗ Attempt {attempt+1}: HTTP {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"    ✗ Attempt {attempt+1}: {str(e)}")
                
                # Wait before retry
                if attempt < 2:
                    time.sleep(2)
            
            if not success:
                print(f"    ✗ Failed to download after 3 attempts")
            
            # Small delay between downloads
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
    
    print(f"\n✅ Downloaded {len(successful_downloads)}/{len(data_urls)} datasets")
    return successful_downloads

def process_json_to_csv():
    """Process downloaded JSON.gz files to CSV format"""
    print("\n📊 Processing JSON files to CSV...")
    
    raw_dir = "data/opensky/raw"
    processed_dir = "data/opensky/processed"
    
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all JSON.gz files
    json_files = [f for f in os.listdir(raw_dir) if f.endswith('.json.gz')]
    
    if not json_files:
        print("No JSON files found to process!")
        return []
    
    print(f"Found {len(json_files)} files to process")
    
    processed_files = []
    
    for i, json_file in enumerate(sorted(json_files), 1):
        json_path = os.path.join(raw_dir, json_file)
        csv_filename = f"dataset_{i:02d}.csv"
        csv_path = os.path.join(processed_dir, csv_filename)
        
        print(f"[{i:02d}/{len(json_files)}] Processing {json_file}...")
        
        try:
            # Read and process the JSON file
            df = process_single_file(json_path)
            
            if df is not None and not df.empty:
                # Save to CSV
                df.to_csv(csv_path, index=False)
                
                processed_files.append({
                    'id': f"dataset_{i:02d}",
                    'original': json_file,
                    'flights': len(df),
                    'csv_path': csv_path,
                    'timestamp': df['timestamp'].iloc[0] if not df.empty else ''
                })
                
                print(f"    ✓ Processed {len(df)} flights")
            else:
                print(f"    ✗ No valid data in file")
                
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
    
    return processed_files

def process_single_file(json_path):
    """Process a single JSON.gz file to DataFrame"""
    try:
        # Read gzipped JSON
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check structure
        if 'states' not in data or 'time' not in data:
            print(f"    Invalid format in {os.path.basename(json_path)}")
            return None
        
        timestamp = data['time']
        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()
        
        flights = []
        
        # Process each state vector
        for state in data['states']:
            if len(state) >= 17:  # Minimum required fields
                flight = process_state_vector(state, timestamp, dt)
                if flight:
                    flights.append(flight)
        
        if not flights:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(flights)
        
        # Add dataset metadata
        df['dataset_source'] = os.path.basename(json_path)
        df['processing_time'] = datetime.now().isoformat()
        
        return df
        
    except Exception as e:
        print(f"    Error processing {os.path.basename(json_path)}: {str(e)}")
        return None

def process_state_vector(state, timestamp, dt):
    """Process a single state vector to dictionary"""
    try:
        # OpenSky state vector format (17 fields)
        # Skip if no position data
        if not state[5] or not state[6]:
            return None
        
        # Get altitude in meters (convert None to 0)
        altitude_m = float(state[7]) if state[7] is not None else 0
        speed_ms = float(state[9]) if state[9] is not None else 0
        
        flight = {
            # Original fields
            'icao24': state[0] or 'N/A',
            'callsign': (state[1] or '').strip(),
            'origin_country': state[2] or 'Unknown',
            'time_position': state[3] or 0,
            'last_contact': state[4] or 0,
            'longitude': float(state[5]) if state[5] else 0,
            'latitude': float(state[6]) if state[6] else 0,
            'baro_altitude': altitude_m,
            'on_ground': bool(state[8]) if state[8] is not None else False,
            'velocity': speed_ms,
            'true_track': float(state[10]) if state[10] else 0,
            'vertical_rate': float(state[11]) if state[11] else 0,
            'sensors': str(state[12]) if state[12] else '[]',
            'geo_altitude': float(state[13]) if state[13] else 0,
            'squawk': state[14] if state[14] else 'N/A',
            'spi': bool(state[15]) if state[15] is not None else False,
            'position_source': state[16] if state[16] else 0,
            
            # Derived fields
            'timestamp': dt.isoformat(),
            'unix_timestamp': timestamp,
            'speed_knots': speed_ms * 1.94384,
            'altitude_ft': altitude_m * 3.28084,
            'vertical_rate_fpm': (float(state[11]) * 196.85) if state[11] else 0,
            
            # Enhanced fields
            'airline': extract_airline((state[1] or '').strip()),
            'origin': infer_airport(float(state[5]) if state[5] else 0, 
                                   float(state[6]) if state[6] else 0, 'origin'),
            'destination': infer_airport(float(state[5]) if state[5] else 0, 
                                        float(state[6]) if state[6] else 0, 'dest'),
            'flight_phase': get_flight_phase(altitude_m, 
                                            bool(state[8]) if state[8] is not None else False),
            'is_anomaly': detect_anomaly(state),
            'anomaly_score': calculate_anomaly_score(state)
        }
        
        return flight
        
    except Exception as e:
        # Skip this flight if there's an error
        return None

def extract_airline(callsign):
    """Extract airline code from callsign"""
    if not callsign or len(callsign) < 2:
        return 'UNK'
    
    # Common airline codes (3-letter)
    airline_codes_3 = ['UAL', 'AAL', 'DAL', 'BAW', 'AFR', 'DLH', 'KLM', 'SWA', 
                      'JBU', 'VIR', 'ACA', 'ANA', 'QFA', 'SIA', 'UAE', 'RYR',
                      'EZY', 'WZZ', 'FDX', 'UPS', 'CKS', 'AVA', 'IBE', 'THY']
    
    # Check 3-letter code
    if len(callsign) >= 3:
        prefix = callsign[:3]
        if prefix in airline_codes_3:
            return prefix
    
    # Check 2-letter codes
    airline_codes_2 = ['LH', 'AF', 'KL', 'AA', 'UA', 'DL', 'BA', 'VS', 'QR', 'EK', 'SQ', 'CX']
    two_letter = callsign[:2]
    if two_letter in airline_codes_2:
        return two_letter
    
    return 'UNK'

def infer_airport(longitude, latitude, airport_type='origin'):
    """Simple airport inference based on coordinates"""
    # Major airports with coordinates
    airports = {
        'JFK': (-73.7781, 40.6413),
        'LAX': (-118.408, 33.9425),
        'LHR': (-0.461389, 51.4775),
        'CDG': (2.547, 49.0097),
        'FRA': (8.570556, 50.033333),
        'DXB': (55.3644, 25.2528),
        'SIN': (103.994, 1.35019),
        'HND': (139.78, 35.5522),
        'ORD': (-87.9048, 41.9786),
        'DFW': (-97.038, 32.8969),
        'ATL': (-84.4281, 33.6367),
        'PEK': (116.597, 40.0725),
        'HKG': (113.915, 22.3089),
        'SYD': (151.177, -33.9461),
        'AMS': (4.76389, 52.3086),
        'MAD': (-3.56676, 40.4936),
        'MUC': (11.7861, 48.3538),
        'YYZ': (-79.6306, 43.6777),
        'GRU': (-46.4731, -23.4356),
        'BOM': (72.8679, 19.0887)
    }
    
    # Find closest airport
    closest = 'UNK'
    min_distance = float('inf')
    
    for code, (lon, lat) in airports.items():
        distance = ((longitude - lon) ** 2 + (latitude - lat) ** 2) ** 0.5
        if distance < min_distance and distance < 5:  # Within 5 degrees
            min_distance = distance
            closest = code
    
    return closest

def get_flight_phase(altitude_m, on_ground):
    """Determine flight phase"""
    if on_ground:
        return 'Ground'
    
    altitude_ft = altitude_m * 3.28084
    
    if altitude_ft < 1000:
        return 'Takeoff/Landing'
    elif altitude_ft < 10000:
        return 'Climb'
    elif altitude_ft < 35000:
        return 'Cruise'
    else:
        return 'High Altitude'

def detect_anomaly(state):
    """Simple anomaly detection"""
    # Check altitude (> 42,000ft or < 300ft)
    if state[7] and (state[7] > 12800 or (state[7] < 100 and not state[8])):
        return True
    
    # Check speed (> 580 knots or < 100 knots)
    if state[9] and (state[9] > 300 or state[9] < 50):
        return True
    
    # Check vertical rate (> 10,000 fpm)
    if state[11] and abs(state[11]) > 50:
        return True
    
    # Check position (invalid coordinates)
    if state[5] and abs(state[5]) > 180:
        return True
    if state[6] and abs(state[6]) > 90:
        return True
    
    return False

def calculate_anomaly_score(state):
    """Calculate anomaly score (0-1)"""
    score = 0.0
    
    # Altitude anomaly (0-0.4)
    if state[7]:
        if state[7] > 12800:  # > 42,000ft
            score += 0.4
        elif state[7] < 100 and not state[8]:  # < 300ft and not on ground
            score += 0.3
    
    # Speed anomaly (0-0.3)
    if state[9]:
        if state[9] > 300:  # > 580 knots
            score += 0.3
        elif state[9] < 50:  # < 100 knots at altitude
            score += 0.2
    
    # Vertical rate anomaly (0-0.2)
    if state[11] and abs(state[11]) > 50:
        score += 0.2
    
    # Position anomaly (0-0.1)
    if state[5] and abs(state[5]) > 180:
        score += 0.1
    if state[6] and abs(state[6]) > 90:
        score += 0.1
    
    return min(score, 1.0)

def create_metadata(processed_files):
    """Create metadata file for all datasets"""
    metadata = {
        'datasets': processed_files,
        'total_datasets': len(processed_files),
        'total_flights': sum(d['flights'] for d in processed_files),
        'processing_date': datetime.now().isoformat(),
        'source': 'OpenSky Network',
        'data_range': '2021-2023',
        'description': 'ADS-B flight data from OpenSky Network',
        'columns_description': {
            'icao24': 'Unique ICAO 24-bit aircraft address',
            'callsign': 'Flight callsign',
            'origin_country': 'Country of origin',
            'longitude': 'Longitude in decimal degrees',
            'latitude': 'Latitude in decimal degrees',
            'altitude_ft': 'Altitude in feet',
            'speed_knots': 'Speed in knots',
            'airline': 'Airline code',
            'origin': 'Inferred origin airport',
            'destination': 'Inferred destination airport',
            'flight_phase': 'Current flight phase',
            'is_anomaly': 'True if anomaly detected',
            'anomaly_score': 'Anomaly score (0-1)',
            'timestamp': 'Timestamp of data'
        }
    }
    
    metadata_path = os.path.join("data/opensky/processed", "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📋 Metadata saved to: {metadata_path}")
    return metadata

def create_sample_data():
    """Create sample data for quick testing"""
    print("\n🎨 Creating sample data for dashboard...")
    
    # Create a small sample from first dataset
    processed_dir = "data/opensky/processed"
    sample_dir = "data/samples"
    
    os.makedirs(sample_dir, exist_ok=True)
    
    # Find first CSV file
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    if csv_files:
        first_csv = os.path.join(processed_dir, sorted(csv_files)[0])
        
        try:
            df = pd.read_csv(first_csv)
            
            # Create sample with first 50 flights
            sample_df = df.head(50)
            sample_path = os.path.join(sample_dir, "sample_flights.csv")
            sample_df.to_csv(sample_path, index=False)
            
            print(f"  ✓ Created sample data: {len(sample_df)} flights")
            
            # Create a JSON version for web
            sample_json = sample_df.to_dict('records')
            json_path = os.path.join(sample_dir, "sample_flights.json")
            with open(json_path, 'w') as f:
                json.dump(sample_json, f, indent=2)
            
            print(f"  ✓ Created JSON version: {json_path}")
            
        except Exception as e:
            print(f"  ✗ Error creating sample: {str(e)}")
    
    return sample_dir

def create_dashboard_snapshot():
    """Create HTML dashboard snapshot"""
    print("\n🖥️ Creating dashboard snapshot...")
    
    # Read sample data if available
    sample_data = []
    sample_path = "data/samples/sample_flights.json"
    
    if os.path.exists(sample_path):
        try:
            with open(sample_path, 'r') as f:
                sample_data = json.load(f)
            print(f"  Loaded {len(sample_data)} sample flights")
        except:
            sample_data = []
    
    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADS-B Flight Dashboard - Snapshot</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #007bff;
        }}
        .header h1 {{
            color: #007bff;
            margin: 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #007bff;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .anomaly-true {{
            color: #dc3545;
            font-weight: bold;
        }}
        .anomaly-false {{
            color: #28a745;
        }}
        .footer {{
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✈️ ADS-B Flight Dashboard Snapshot</h1>
            <p>OpenSky Network Data | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="total-flights">{len(sample_data)}</div>
                <div class="stat-label">Total Flights</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="anomaly-count">{sum(1 for f in sample_data if f.get('is_anomaly', False))}</div>
                <div class="stat-label">Anomalies</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-altitude">{np.mean([f.get('altitude_ft', 0) for f in sample_data]) if sample_data else 0:.0f}</div>
                <div class="stat-label">Avg Altitude (ft)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-speed">{np.mean([f.get('speed_knots', 0) for f in sample_data]) if sample_data else 0:.0f}</div>
                <div class="stat-label">Avg Speed (knots)</div>
            </div>
        </div>
        
        <h2>Flight Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Callsign</th>
                    <th>Airline</th>
                    <th>Route</th>
                    <th>Altitude (ft)</th>
                    <th>Speed (knots)</th>
                    <th>Anomaly</th>
                </tr>
            </thead>
            <tbody>
                {"".join([f'''
                <tr>
                    <td>{flight.get('callsign', 'N/A')}</td>
                    <td>{flight.get('airline', 'UNK')}</td>
                    <td>{flight.get('origin', 'UNK')} → {flight.get('destination', 'UNK')}</td>
                    <td>{flight.get('altitude_ft', 0):.0f}</td>
                    <td>{flight.get('speed_knots', 0):.0f}</td>
                    <td class="{'anomaly-true' if flight.get('is_anomaly', False) else 'anomaly-false'}">
                        {'✓' if flight.get('is_anomaly', False) else '✗'}
                    </td>
                </tr>
                ''' for flight in sample_data[:20]])}
            </tbody>
        </table>
        
        <div class="footer">
            <p>Data Source: OpenSky Network | Processed: {datetime.now().strftime('%Y-%m-%d')}</p>
            <p>This is a static snapshot of flight data. For live updates, run the full dashboard.</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML
    snapshot_path = os.path.join("data/opensky/processed", "dashboard_snapshot.html")
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ Created dashboard snapshot: {snapshot_path}")
    return snapshot_path

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 OpenSky Data Pipeline")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Step 1: Download data
        downloaded = download_opensky_data()
        
        if not downloaded:
            print("\n❌ No data downloaded. Trying alternative approach...")
            # Try with smaller dataset
            create_fallback_data()
            return
        
        # Step 2: Process to CSV
        processed = process_json_to_csv()
        
        if not processed:
            print("\n❌ No data processed.")
            return
        
        # Step 3: Create metadata
        metadata = create_metadata(processed)
        
        # Step 4: Create sample data
        create_sample_data()
        
        # Step 5: Create dashboard snapshot
        create_dashboard_snapshot()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("✅ Pipeline Completed Successfully!")
        print("=" * 70)
        
        print(f"\n📊 Summary:")
        print(f"  • Downloaded datasets: {len(downloaded)}")
        print(f"  • Processed datasets: {len(processed)}")
        print(f"  • Total flights: {metadata['total_flights']:,}")
        print(f"  • Time elapsed: {elapsed:.1f} seconds")
        
        print(f"\n📁 Files created:")
        print(f"  • Raw data: data/opensky/raw/")
        print(f"  • Processed CSV: data/opensky/processed/")
        print(f"  • Metadata: data/opensky/processed/metadata.json")
        print(f"  • Dashboard snapshot: data/opensky/processed/dashboard_snapshot.html")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Check the data: python -c \"import pandas as pd; df=pd.read_csv('data/opensky/processed/dataset_01.csv'); print(f'Dataset has {len(df)} flights')\"")
        print(f"  2. Open dashboard snapshot in browser")
        print(f"  3. Update your app to use these datasets")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        # Create fallback data
        create_fallback_data()

def create_fallback_data():
    """Create fallback data if download fails"""
    print("\n🔄 Creating fallback datasets...")
    
    os.makedirs("data/opensky/processed", exist_ok=True)
    
    # Create 30 sample datasets
    for i in range(1, 31):
        df = create_sample_dataset(i)
        csv_path = os.path.join("data/opensky/processed", f"dataset_{i:02d}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Created dataset_{i:02d}.csv with {len(df)} flights")
    
    # Create metadata
    metadata = {
        'datasets': [
            {
                'id': f"dataset_{i:02d}",
                'flights': 100,
                'type': 'simulated',
                'description': 'Simulated ADS-B flight data'
            }
            for i in range(1, 31)
        ],
        'total_datasets': 30,
        'total_flights': 3000,
        'source': 'Simulated data (fallback)'
    }
    
    metadata_path = os.path.join("data/opensky/processed", "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Created 30 simulated datasets")
    print(f"📁 Data available at: data/opensky/processed/")

def create_sample_dataset(dataset_id):
    """Create a sample dataset with realistic flight data"""
    flights = []
    
    airlines = ['UAL', 'AAL', 'DAL', 'BAW', 'AFR', 'DLH', 'KLM', 'SWA']
    airports = ['JFK', 'LAX', 'LHR', 'CDG', 'FRA', 'DXB', 'SIN', 'HND', 'ORD', 'DFW']
    
    for j in range(100):
        airline = np.random.choice(airlines)
        flight_num = f"{np.random.randint(100, 9999)}"
        
        flight = {
            'icao24': f"a{np.random.randint(100000, 999999):06x}",
            'callsign': f"{airline}{flight_num}",
            'origin_country': 'United States',
            'longitude': np.random.uniform(-180, 180),
            'latitude': np.random.uniform(-90, 90),
            'altitude_ft': np.random.uniform(10000, 40000),
            'speed_knots': np.random.uniform(300, 550),
            'airline': airline,
            'origin': np.random.choice(airports),
            'destination': np.random.choice([a for a in airports if a != airline]),
            'flight_phase': np.random.choice(['Climb', 'Cruise', 'Descent']),
            'is_anomaly': np.random.random() < 0.1,
            'anomaly_score': np.random.uniform(0, 0.3) if np.random.random() < 0.1 else 0,
            'timestamp': f"2023-01-{dataset_id:02d}T{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:00"
        }
        flights.append(flight)
    
    return pd.DataFrame(flights)

if __name__ == "__main__":
    # Install required packages if not present
    try:
        import requests
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'requests', 'pandas', 'numpy'])
    
    main()