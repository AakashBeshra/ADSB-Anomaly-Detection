# generate_opensky_datasets.py
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import random

def generate_realistic_flight_data(dataset_id, num_flights=500):
    """Generate realistic flight data similar to OpenSky format"""
    
    print(f"Generating dataset {dataset_id:02d}...")
    
    # Real airline codes and information
    airlines = [
        {'code': 'UAL', 'name': 'United Airlines', 'country': 'United States'},
        {'code': 'AAL', 'name': 'American Airlines', 'country': 'United States'},
        {'code': 'DAL', 'name': 'Delta Air Lines', 'country': 'United States'},
        {'code': 'BAW', 'name': 'British Airways', 'country': 'United Kingdom'},
        {'code': 'AFR', 'name': 'Air France', 'country': 'France'},
        {'code': 'DLH', 'name': 'Lufthansa', 'country': 'Germany'},
        {'code': 'KLM', 'name': 'KLM Royal Dutch', 'country': 'Netherlands'},
        {'code': 'SIA', 'name': 'Singapore Airlines', 'country': 'Singapore'},
        {'code': 'ANA', 'name': 'All Nippon Airways', 'country': 'Japan'},
        {'code': 'QFA', 'name': 'Qantas', 'country': 'Australia'},
        {'code': 'UAE', 'name': 'Emirates', 'country': 'UAE'},
        {'code': 'CCA', 'name': 'Air China', 'country': 'China'},
        {'code': 'JAL', 'name': 'Japan Airlines', 'country': 'Japan'},
        {'code': 'THY', 'name': 'Turkish Airlines', 'country': 'Turkey'},
        {'code': 'CPA', 'name': 'Cathay Pacific', 'country': 'Hong Kong'},
        {'code': 'VIR', 'name': 'Virgin Atlantic', 'country': 'United Kingdom'},
        {'code': 'SWA', 'name': 'Southwest Airlines', 'country': 'United States'},
        {'code': 'JBU', 'name': 'JetBlue', 'country': 'United States'},
        {'code': 'RYR', 'name': 'Ryanair', 'country': 'Ireland'},
        {'code': 'EZY', 'name': 'EasyJet', 'country': 'United Kingdom'}
    ]
    
    # Major airport pairs (real routes)
    routes = [
        # Transatlantic
        ('JFK', 'LHR'), ('JFK', 'CDG'), ('JFK', 'FRA'), ('JFK', 'AMS'),
        ('LAX', 'LHR'), ('LAX', 'CDG'), ('LAX', 'FRA'), ('LAX', 'NRT'),
        ('ORD', 'LHR'), ('ORD', 'CDG'), ('ORD', 'FRA'), ('ORD', 'MUC'),
        ('DFW', 'LHR'), ('DFW', 'CDG'), ('DFW', 'FRA'), ('DFW', 'AMS'),
        
        # Transpacific
        ('LAX', 'NRT'), ('LAX', 'HND'), ('LAX', 'ICN'), ('LAX', 'PVG'),
        ('SFO', 'NRT'), ('SFO', 'HND'), ('SFO', 'ICN'), ('SFO', 'PEK'),
        ('JFK', 'NRT'), ('JFK', 'HND'), ('JFK', 'ICN'), ('JFK', 'PEK'),
        
        # Domestic US
        ('JFK', 'LAX'), ('JFK', 'SFO'), ('JFK', 'MIA'), ('JFK', 'ORD'),
        ('LAX', 'ORD'), ('LAX', 'DFW'), ('LAX', 'ATL'), ('LAX', 'MIA'),
        ('ORD', 'DFW'), ('ORD', 'ATL'), ('ORD', 'DEN'), ('ORD', 'LAS'),
        ('ATL', 'DFW'), ('ATL', 'DEN'), ('ATL', 'LAS'), ('ATL', 'SEA'),
        
        # Europe
        ('LHR', 'CDG'), ('LHR', 'FRA'), ('LHR', 'AMS'), ('LHR', 'MAD'),
        ('CDG', 'FRA'), ('CDG', 'AMS'), ('CDG', 'MUC'), ('CDG', 'ZRH'),
        ('FRA', 'AMS'), ('FRA', 'MUC'), ('FRA', 'VIE'), ('FRA', 'ZRH'),
        
        # Asia
        ('HND', 'ICN'), ('HND', 'PVG'), ('HND', 'PEK'), ('HND', 'SIN'),
        ('ICN', 'PVG'), ('ICN', 'PEK'), ('ICN', 'SIN'), ('ICN', 'BKK'),
        ('SIN', 'BKK'), ('SIN', 'KUL'), ('SIN', 'HKG'), ('SIN', 'DEL'),
        
        # Middle East
        ('DXB', 'LHR'), ('DXB', 'CDG'), ('DXB', 'FRA'), ('DXB', 'JFK'),
        ('DXB', 'LAX'), ('DXB', 'SIN'), ('DXB', 'BOM'), ('DXB', 'DEL'),
        
        # Australia
        ('SYD', 'LAX'), ('SYD', 'SFO'), ('SYD', 'DXB'), ('SYD', 'SIN'),
        ('MEL', 'LAX'), ('MEL', 'SFO'), ('MEL', 'DXB'), ('MEL', 'SIN')
    ]
    
    # Airport coordinates
    airport_coords = {
        'JFK': (-73.7781, 40.6413, 'New York'),
        'LAX': (-118.408, 33.9425, 'Los Angeles'),
        'LHR': (-0.461389, 51.4775, 'London'),
        'CDG': (2.547, 49.0097, 'Paris'),
        'FRA': (8.570556, 50.033333, 'Frankfurt'),
        'AMS': (4.76389, 52.3086, 'Amsterdam'),
        'NRT': (140.386, 35.7647, 'Tokyo'),
        'HND': (139.78, 35.5522, 'Tokyo'),
        'ICN': (126.451, 37.4602, 'Seoul'),
        'PVG': (121.805, 31.144, 'Shanghai'),
        'PEK': (116.597, 40.0725, 'Beijing'),
        'SIN': (103.994, 1.35019, 'Singapore'),
        'DXB': (55.3644, 25.2528, 'Dubai'),
        'SYD': (151.177, -33.9461, 'Sydney'),
        'ORD': (-87.9048, 41.9786, 'Chicago'),
        'DFW': (-97.038, 32.8969, 'Dallas'),
        'ATL': (-84.4281, 33.6367, 'Atlanta'),
        'MIA': (-80.2903, 25.7933, 'Miami'),
        'SFO': (-122.375, 37.619, 'San Francisco'),
        'SEA': (-122.309, 47.449, 'Seattle'),
        'DEN': (-104.673, 39.8617, 'Denver'),
        'LAS': (-115.152, 36.08, 'Las Vegas'),
        'MUC': (11.7861, 48.3538, 'Munich'),
        'ZRH': (8.54917, 47.4647, 'Zurich'),
        'MAD': (-3.56676, 40.4936, 'Madrid'),
        'BKK': (100.747, 13.9125, 'Bangkok'),
        'HKG': (113.915, 22.3089, 'Hong Kong'),
        'DEL': (77.1031, 28.5665, 'Delhi'),
        'BOM': (72.8679, 19.0887, 'Mumbai'),
        'MEL': (144.843, -37.6733, 'Melbourne')
    }
    
    flights = []
    
    # Base timestamp for this dataset
    base_date = datetime(2023, 1, 1) + timedelta(days=dataset_id-1)
    
    for i in range(num_flights):
        # Select random airline
        airline = random.choice(airlines)
        flight_num = f"{random.randint(100, 9999)}"
        callsign = f"{airline['code']}{flight_num}"
        
        # Select random route
        origin, destination = random.choice(routes)
        
        # Get coordinates for origin and destination
        origin_lon, origin_lat, origin_city = airport_coords.get(origin, (0, 0, 'Unknown'))
        dest_lon, dest_lat, dest_city = airport_coords.get(destination, (0, 0, 'Unknown'))
        
        # Generate flight time (simulate flight progress)
        flight_progress = random.uniform(0.1, 0.9)  # 10% to 90% complete
        
        # Calculate current position along great circle route
        current_lon = origin_lon + (dest_lon - origin_lon) * flight_progress
        current_lat = origin_lat + (dest_lat - origin_lat) * flight_progress
        
        # Add some randomness to position
        current_lon += random.uniform(-2, 2)
        current_lat += random.uniform(-1, 1)
        
        # Generate realistic flight parameters
        altitude_ft = random.choices(
            [28000, 31000, 33000, 35000, 37000, 39000, 41000],
            weights=[5, 15, 25, 25, 15, 10, 5]
        )[0]
        
        speed_knots = random.choices(
            [420, 450, 480, 500, 520, 550],
            weights=[5, 15, 30, 30, 15, 5]
        )[0]
        
        # Generate timestamp for this flight
        flight_time = base_date + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Create flight record
        flight = {
            # OpenSky-like fields
            'icao24': f"{random.choice(['a', 'b', 'c', 'd', 'e', 'f'])}{random.randint(100000, 999999):06x}",
            'callsign': callsign,
            'origin_country': airline['country'],
            'time_position': int(flight_time.timestamp()),
            'last_contact': int(flight_time.timestamp()),
            'longitude': current_lon,
            'latitude': current_lat,
            'baro_altitude': altitude_ft / 3.28084,  # Convert to meters
            'on_ground': False,
            'velocity': speed_knots / 1.94384,  # Convert to m/s
            'true_track': random.uniform(0, 360),
            'vertical_rate': random.uniform(-10, 10),  # m/s
            'geo_altitude': altitude_ft / 3.28084,
            'squawk': f"{random.randint(1000, 7777)}",
            'spi': False,
            'position_source': 0,
            
            # Enhanced fields
            'airline': airline['code'],
            'airline_name': airline['name'],
            'origin': origin,
            'origin_city': origin_city,
            'destination': destination,
            'destination_city': dest_city,
            'flight_number': flight_num,
            'altitude_ft': altitude_ft,
            'speed_knots': speed_knots,
            'vertical_rate_fpm': random.uniform(-2000, 2000),
            'timestamp': flight_time.isoformat(),
            'flight_progress': flight_progress,
            'dataset_id': f"dataset_{dataset_id:02d}",
            
            # Flight phase based on altitude
            'flight_phase': get_flight_phase(altitude_ft, flight_progress),
            
            # Anomaly detection fields
            'is_anomaly': False,
            'anomaly_type': 'none',
            'anomaly_score': 0.0,
            'anomaly_reason': ''
        }
        
        # Add anomalies for 10% of flights
        if random.random() < 0.1:
            flight = add_anomaly(flight)
        
        flights.append(flight)
    
    return pd.DataFrame(flights)

def get_flight_phase(altitude_ft, progress):
    """Determine flight phase"""
    if progress < 0.1:
        return 'Takeoff'
    elif progress > 0.9:
        return 'Landing'
    elif altitude_ft < 10000:
        return 'Climb'
    elif altitude_ft > 35000:
        return 'High Cruise'
    else:
        return 'Cruise'

def add_anomaly(flight):
    """Add realistic anomaly to flight"""
    anomaly_types = [
        ('altitude', 'Unusual altitude for flight phase'),
        ('speed', 'Speed deviation from normal'),
        ('vertical_rate', 'Extreme vertical rate'),
        ('position', 'Unusual position or route deviation'),
        ('communication', 'Loss of communication'),
        ('track', 'Unusual track angle')
    ]
    
    anomaly_type, reason = random.choice(anomaly_types)
    
    flight['is_anomaly'] = True
    flight['anomaly_type'] = anomaly_type
    flight['anomaly_reason'] = reason
    
    # Generate anomaly score (0.1 to 0.9)
    flight['anomaly_score'] = random.uniform(0.3, 0.9)
    
    # Apply specific anomaly based on type
    if anomaly_type == 'altitude':
        flight['altitude_ft'] = random.choice([15000, 45000])  # Too low or too high
        flight['baro_altitude'] = flight['altitude_ft'] / 3.28084
        flight['geo_altitude'] = flight['altitude_ft'] / 3.28084
        
    elif anomaly_type == 'speed':
        flight['speed_knots'] = random.choice([250, 650])  # Too slow or too fast
        flight['velocity'] = flight['speed_knots'] / 1.94384
        
    elif anomaly_type == 'vertical_rate':
        flight['vertical_rate'] = random.choice([-25, 25])  # Extreme climb/descent
        flight['vertical_rate_fpm'] = flight['vertical_rate'] * 196.85
        
    elif anomaly_type == 'position':
        # Move to unusual location
        flight['longitude'] += random.uniform(-20, 20)
        flight['latitude'] += random.uniform(-10, 10)
        
    elif anomaly_type == 'communication':
        flight['squawk'] = '7500'  # Hijack code (for testing)
        
    elif anomaly_type == 'track':
        flight['true_track'] = (flight['true_track'] + 180) % 360  # Reverse direction
    
    return flight

def generate_all_datasets(num_datasets=30, output_dir="data/opensky"):
    """Generate all datasets"""
    
    print(f"🚀 Generating {num_datasets} realistic OpenSky datasets...")
    
    # Create directories
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    all_datasets = []
    total_flights = 0
    
    for dataset_id in range(1, num_datasets + 1):
        # Generate variable number of flights per dataset
        num_flights = random.randint(400, 800)
        
        # Generate dataset
        df = generate_realistic_flight_data(dataset_id, num_flights)
        
        # Save as CSV (processed format)
        csv_filename = f"dataset_{dataset_id:02d}.csv"
        csv_path = os.path.join(processed_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        # Create sample (first 50 flights) for quick testing
        sample_df = df.head(50)
        sample_path = os.path.join(processed_dir, f"sample_{dataset_id:02d}.csv")
        sample_df.to_csv(sample_path, index=False)
        
        # Create JSON version (raw format simulation)
        json_filename = f"dataset_{dataset_id:02d}.json"
        json_path = os.path.join(raw_dir, json_filename)
        
        # Convert to OpenSky-like JSON format
        json_data = convert_to_opensky_format(df)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        dataset_info = {
            'id': f"dataset_{dataset_id:02d}",
            'name': f"OpenSky Simulation - Day {dataset_id}",
            'date': f"2023-01-{dataset_id:02d}",
            'flights': len(df),
            'anomalies': df['is_anomaly'].sum(),
            'csv_path': csv_path,
            'json_path': json_path,
            'sample_path': sample_path
        }
        
        all_datasets.append(dataset_info)
        total_flights += len(df)
        
        print(f"  ✓ Dataset {dataset_id:02d}: {len(df):,} flights ({df['is_anomaly'].sum()} anomalies)")
    
    # Create metadata
    metadata = {
        'datasets': all_datasets,
        'total_datasets': num_datasets,
        'total_flights': total_flights,
        'total_anomalies': sum(d['anomalies'] for d in all_datasets),
        'generated_date': datetime.now().isoformat(),
        'description': 'Realistic OpenSky-style flight datasets for ADS-B anomaly detection',
        'airlines_represented': list(set([d['csv_path'].split('/')[-1][:3] for d in all_datasets])),
        'airports_represented': list(set([
            f"{row['origin']}-{row['destination']}" 
            for dataset in all_datasets 
            for _, row in pd.read_csv(dataset['csv_path']).iterrows()
        ]))[:20],  # First 20 unique routes
        'anomaly_types': ['altitude', 'speed', 'vertical_rate', 'position', 'communication', 'track'],
        'data_quality': 'High - Simulated but realistic flight data'
    }
    
    metadata_path = os.path.join(processed_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create passenger data
    passenger_data = generate_passenger_data(all_datasets)
    passenger_path = os.path.join(processed_dir, "passenger_manifest.csv")
    passenger_data.to_csv(passenger_path, index=False)
    
    # Create dashboard snapshot
    create_dashboard_snapshot(all_datasets, processed_dir)
    
    print(f"\n✅ Generation complete!")
    print(f"📊 Summary:")
    print(f"  • Datasets: {num_datasets}")
    print(f"  • Total flights: {total_flights:,}")
    print(f"  • Anomalies: {metadata['total_anomalies']} ({metadata['total_anomalies']/total_flights*100:.1f}%)")
    print(f"  • Airlines: {len(metadata['airlines_represented'])}")
    print(f"  • Routes: {len(metadata['airports_represented'])}")
    
    print(f"\n📁 Files created:")
    print(f"  • Raw JSON: {raw_dir}/")
    print(f"  • Processed CSV: {processed_dir}/")
    print(f"  • Metadata: {metadata_path}")
    print(f"  • Passenger data: {passenger_path}")
    print(f"  • Dashboard snapshot: {processed_dir}/dashboard_snapshot.html")
    
    return metadata

def convert_to_opensky_format(df):
    """Convert DataFrame to OpenSky-like JSON format"""
    states = []
    
    for _, row in df.iterrows():
        state = [
            row['icao24'],                    # 0: icao24
            row['callsign'],                  # 1: callsign
            row['origin_country'],            # 2: origin_country
            row['time_position'],             # 3: time_position
            row['last_contact'],              # 4: last_contact
            row['longitude'],                 # 5: longitude
            row['latitude'],                  # 6: latitude
            row['baro_altitude'],             # 7: baro_altitude
            row['on_ground'],                 # 8: on_ground
            row['velocity'],                  # 9: velocity
            row['true_track'],                # 10: true_track
            row['vertical_rate'],             # 11: vertical_rate
            [],                               # 12: sensors
            row['geo_altitude'],              # 13: geo_altitude
            row['squawk'],                    # 14: squawk
            row['spi'],                       # 15: spi
            row['position_source']            # 16: position_source
        ]
        states.append(state)
    
    return {
        'time': int(datetime.now().timestamp()),
        'states': states
    }

def generate_passenger_data(datasets_info, num_passengers=10000):
    """Generate synthetic passenger data"""
    print("\n👥 Generating passenger data...")
    
    # Collect all flight callsigns from datasets
    all_flights = []
    for dataset in datasets_info:
        df = pd.read_csv(dataset['csv_path'])
        flight_callsigns = df['callsign'].unique().tolist()
        all_flights.extend(flight_callsigns)
    
    # Unique flights
    unique_flights = list(set(all_flights))[:200]  # Limit to 200 unique flights
    
    # Generate passenger data
    passengers = []
    
    first_names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda', 
                  'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
                  'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher', 'Nancy', 'Daniel', 'Lisa',
                  'Matthew', 'Margaret', 'Anthony', 'Betty', 'Donald', 'Sandra']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
                 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson']
    
    countries = ['United States', 'United Kingdom', 'Canada', 'Australia', 'Germany', 'France',
                'Japan', 'China', 'India', 'Brazil', 'Mexico', 'Spain', 'Italy', 'Netherlands',
                'South Korea', 'Singapore', 'UAE', 'Switzerland', 'Sweden', 'Norway']
    
    for i in range(num_passengers):
        passenger_id = f"PAX{i+1:05d}"
        flight_callsign = random.choice(unique_flights)
        
        passenger = {
            'passenger_id': passenger_id,
            'flight_callsign': flight_callsign,
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names),
            'full_name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'nationality': random.choice(countries),
            'seat': f"{random.randint(1, 40)}{random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}",
            'booking_class': random.choice(['Economy', 'Premium Economy', 'Business', 'First']),
            'status': random.choice(['Checked-in', 'Boarded', 'Scheduled', 'No-show']),
            'check_in_time': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat(),
            'baggage_count': random.randint(0, 3),
            'special_requests': random.choice(['None', 'Wheelchair', 'Vegetarian Meal', 'Extra Legroom', 'VIP']),
            'frequent_flyer_tier': random.choice(['None', 'Silver', 'Gold', 'Platinum']),
            'ticket_price': round(random.uniform(200, 5000), 2)
        }
        
        passengers.append(passenger)
    
    print(f"  ✓ Generated {len(passengers)} passenger records")
    return pd.DataFrame(passengers)

def create_dashboard_snapshot(datasets_info, output_dir):
    """Create HTML dashboard snapshot"""
    print("\n🎨 Creating dashboard snapshot...")
    
    # Read first dataset for sample data
    first_dataset = datasets_info[0]
    df = pd.read_csv(first_dataset['csv_path'])
    
    # Create HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADS-B Flight Dashboard - Realistic OpenSky Data</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #667eea;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.8rem;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.2rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 1rem;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{
            background: #f5f7ff;
        }}
        .anomaly-true {{ color: #e74c3c; font-weight: bold; }}
        .anomaly-false {{ color: #27ae60; }}
        .passenger-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }}
        .passenger-id {{
            font-weight: bold;
            color: #667eea;
        }}
        .dataset-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
        }}
        @media (max-width: 1200px) {{
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .dashboard {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: 1fr; }}
            .container {{ padding: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✈️ ADS-B Flight Dashboard</h1>
            <p>Realistic OpenSky Flight Data & Anomaly Detection</p>
            <p style="color: #667eea; margin-top: 10px;">30 Simulated Datasets | {len(datasets_info)} Flight Records</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{len(datasets_info)}</div>
                <div class="stat-label">Datasets</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(d['flights'] for d in datasets_info):,}</div>
                <div class="stat-label">Total Flights</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(d['anomalies'] for d in datasets_info)}</div>
                <div class="stat-label">Anomalies Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(set([d['csv_path'].split('/')[-1][:3] for d in datasets_info]))}</div>
                <div class="stat-label">Airlines</div>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h2>🛫 Active Flights</h2>
                <div style="height: 400px; background: #f8f9fa; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 4rem; color: #667eea;">🌍</div>
                        <p style="font-size: 1.2rem; color: #666;">Interactive Flight Map</p>
                        <p style="color: #999;">Real-time flight positions on global map</p>
                    </div>
                </div>
                
                <h3>Flight Status Table</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Flight</th>
                            <th>Route</th>
                            <th>Altitude</th>
                            <th>Speed</th>
                            <th>Anomaly</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f'''
                        <tr>
                            <td><strong>{row['callsign']}</strong><br><small>{row['airline_name']}</small></td>
                            <td>{row['origin']} → {row['destination']}<br><small>{row['origin_city']} to {row['destination_city']}</small></td>
                            <td>{int(row['altitude_ft']):,} ft</td>
                            <td>{int(row['speed_knots'])} knots</td>
                            <td class="{'anomaly-true' if row['is_anomaly'] else 'anomaly-false'}">
                                {'⚠️' if row['is_anomaly'] else '✓'}
                            </td>
                        </tr>
                        ''' for _, row in df.head(8).iterrows()])}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>📊 Anomaly Detection</h2>
                <div style="height: 200px; background: #f8f9fa; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                    <div style="text-align: center;">
                        <div style="font-size: 3rem; color: #e74c3c;">📈</div>
                        <p style="color: #666;">Anomaly Detection Analytics</p>
                    </div>
                </div>
                
                <h3>Passenger Manifest</h3>
                <div style="max-height: 300px; overflow-y: auto;">
                    {"".join([f'''
                    <div class="passenger-card">
                        <div class="passenger-id">PAX{i:05d}</div>
                        <div style="margin-top: 5px;">
                            <strong>Passenger {i}</strong><br>
                            Flight: FLT{i:03d} | Seat: {random.randint(1, 40)}{random.choice(['A', 'B', 'C'])}<br>
                            Status: <span style="color: #27ae60;">Checked-in</span>
                        </div>
                    </div>
                    ''' for i in range(1, 6)])}
                </div>
            </div>
        </div>
        
        <div class="dataset-info">
            <h3>📋 Dataset Information</h3>
            <p>This dashboard displays realistic flight data generated in OpenSky format, including:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>30 datasets</strong> with varying flight conditions</li>
                <li><strong>Real airline codes and routes</strong> (UAL, AAL, BAW, etc.)</li>
                <li><strong>Anomaly detection</strong> with 6 different anomaly types</li>
                <li><strong>Passenger manifests</strong> for each flight</li>
                <li><strong>Real-time simulation</strong> of flight progress</li>
            </ul>
            <p style="margin-top: 15px;"><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="footer">
            <p>🚀 ADS-B Anomaly Detection System | Realistic OpenSky Data Simulation</p>
            <p>For live updates, run the full dashboard application</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML file
    snapshot_path = os.path.join(output_dir, "dashboard_snapshot.html")
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✓ Created dashboard snapshot: {snapshot_path}")
    return snapshot_path

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 Realistic OpenSky Dataset Generator")
    print("=" * 70)
    print("\nThis script will generate 30 realistic OpenSky-style datasets")
    print("with flight data, passenger manifests, and a dashboard snapshot.")
    print("\nFeatures:")
    print("  • 30 datasets with 400-800 flights each")
    print("  • Real airline codes and routes")
    print("  • 10% anomaly rate for testing")
    print("  • Passenger manifests")
    print("  • Dashboard HTML snapshot")
    print("=" * 70)
    
    try:
        # Generate datasets
        metadata = generate_all_datasets(30, "data/opensky")
        
        print("\n" + "=" * 70)
        print("✅ Generation Complete!")
        print("=" * 70)
        
        print(f"\n🚀 Your datasets are ready!")
        print(f"\n📁 Directory structure:")
        print(f"  data/opensky/")
        print(f"  ├── raw/              # JSON format datasets")
        print(f"  ├── processed/        # CSV format datasets (30 files)")
        print(f"  │   ├── dataset_01.csv")
        print(f"  │   ├── dataset_02.csv")
        print(f"  │   ├── ...")
        print(f"  │   ├── metadata.json")
        print(f"  │   ├── passenger_manifest.csv")
        print(f"  │   └── dashboard_snapshot.html")
        print(f"  └── samples/          # Sample data")
        
        print(f"\n🔧 Quick test:")
        print(f"  python -c \"import pandas as pd; df=pd.read_csv('data/opensky/processed/dataset_01.csv'); print(f'✓ Dataset has {{len(df)}} flights, {{df.is_anomaly.sum()}} anomalies')\"")
        
        print(f"\n🌐 Open dashboard snapshot in browser:")
        print(f"  Open: data/opensky/processed/dashboard_snapshot.html")
        
        print(f"\n📊 Sample query:")
        print(f'''  import pandas as pd
  df = pd.read_csv('data/opensky/processed/dataset_01.csv')
  print(f"Total flights: {{len(df)}}")
  print(f"Anomalies: {{df['is_anomaly'].sum()}}")
  print(f"Airlines: {{df['airline'].unique()[:5]}}")''')
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check and install required packages
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pandas', 'numpy'])
    
    main()