# direct_download.py (in root folder)
import requests
import os
from pathlib import Path

def setup_project_structure():
    """Create all necessary folders"""
    
    project_root = Path(__file__).parent  # Gets root folder
    
    # Data directories
    data_dirs = [
        "data/opensky",           # For raw downloaded files
        "data/opensky_processed", # For CSV datasets
        "data/passenger",         # For passenger data
        "data/backups"            # Optional backups
    ]
    
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}/")
    
    return project_root

def download_opensky_datasets(project_root):
    """Download 30 OpenSky datasets"""
    
    # Create data directory paths
    raw_data_dir = project_root / "data" / "opensky"
    processed_data_dir = project_root / "data" / "opensky_processed"
    
    # List of 30 state vector files
    state_files = [
        # [Your 30 URLs here - same as before]
        "https://s3.opensky-network.org/data-samples/states/2023-01-01-00.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-01.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-02.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-03.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-04.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-05.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-06.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-07.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-08.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-09.json.gz",
    
    # Different days for variety
    "https://s3.opensky-network.org/data-samples/states/2023-01-15-12.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-15-13.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-15-14.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-15-15.json.gz",
    
    # 2022 data
    "https://s3.opensky-network.org/data-samples/states/2022-06-01-10.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2022-06-01-11.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2022-06-01-12.json.gz",
    
    # 2021 data
    "https://s3.opensky-network.org/data-samples/states/2021-12-25-18.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2021-12-25-19.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2021-12-25-20.json.gz",
    
    # Mix of different times
    "https://s3.opensky-network.org/data-samples/states/2023-04-10-08.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-04-10-09.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-07-20-16.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-07-20-17.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-10-05-14.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-10-05-15.json.gz",
    
    # Busy airport times
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-20.json.gz",  # Evening
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-21.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-22.json.gz",
    "https://s3.opensky-network.org/data-samples/states/2023-01-01-23.json.gz",
    ]
    
    print(f"Downloading {len(state_files)} datasets from OpenSky...")
    
    for i, url in enumerate(state_files, 1):
        try:
            filename = url.split('/')[-1]
            print(f"[{i:02d}/{len(state_files)}] Downloading {filename}...")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save to data/opensky/
                filepath = raw_data_dir / f"dataset_{i:02d}.json.gz"
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                size_mb = len(response.content) / 1024 / 1024
                print(f"  ✓ Saved to data/opensky/dataset_{i:02d}.json.gz ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ Failed: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    return raw_data_dir

def generate_passenger_data(project_root, num_passengers=10000):
    """Generate synthetic passenger data if not exists"""
    
    passenger_dir = project_root / "data" / "passenger"
    
    # Check if passenger data already exists
    passenger_files = list(passenger_dir.glob("*.csv"))
    
    if passenger_files:
        print(f"✓ Passenger data already exists: {len(passenger_files)} files")
        return passenger_dir
    
    # Generate passenger data
    import pandas as pd
    from faker import Faker
    import random
    
    print("Generating passenger data...")
    
    fake = Faker()
    
    # Generate flight callsigns that match OpenSky data
    airlines = ['UAL', 'DAL', 'AAL', 'BAW', 'AFR', 'DLH', 'KLM', 'SWA']
    flight_callsigns = []
    
    for _ in range(500):  # 500 different flights
        airline = random.choice(airlines)
        flight_num = f"{random.randint(100, 9999)}"
        callsign = f"{airline}{flight_num}"
        flight_callsigns.append(callsign)
    
    # Generate passengers
    passengers = []
    for i in range(num_passengers):
        passenger = {
            'passenger_id': f"PAX{i:05d}",
            'flight_callsign': random.choice(flight_callsigns),
            'name': fake.name(),
            'nationality': fake.country(),
            'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            'seat': f"{random.randint(1, 40)}{random.choice('ABCDEF')}",
            'booking_class': random.choice(['Economy', 'Premium Economy', 'Business', 'First']),
            'status': random.choice(['Checked-in', 'Boarded', 'Scheduled', 'No-show']),
            'check_in_time': fake.date_time_this_month().isoformat(),
            'luggage_count': random.randint(0, 3),
            'special_assistance': random.choice(['None', 'Wheelchair', 'Unaccompanied Minor', 'Pet']),
            'frequent_flyer_tier': random.choice(['None', 'Silver', 'Gold', 'Platinum'])
        }
        passengers.append(passenger)
    
    # Save to CSV
    df_passengers = pd.DataFrame(passengers)
    passenger_file = passenger_dir / "passenger_manifest.csv"
    df_passengers.to_csv(passenger_file, index=False)
    
    print(f"✓ Generated {num_passengers} passenger records")
    print(f"  Saved to: data/passenger/passenger_manifest.csv")
    
    # Also create a flight manifest
    flights = []
    for callsign in flight_callsigns[:100]:  # First 100 flights
        flight = {
            'flight_callsign': callsign,
            'airline': callsign[:3],
            'origin': random.choice(['JFK', 'LAX', 'LHR', 'CDG', 'FRA', 'DXB', 'SIN', 'HND']),
            'destination': random.choice(['JFK', 'LAX', 'LHR', 'CDG', 'FRA', 'DXB', 'SIN', 'HND']),
            'scheduled_departure': fake.date_time_this_month().isoformat(),
            'scheduled_arrival': fake.date_time_this_month().isoformat(),
            'aircraft_type': random.choice(['B737', 'A320', 'B777', 'A350', 'B787']),
            'capacity': random.choice([180, 200, 250, 300, 350])
        }
        flights.append(flight)
    
    df_flights = pd.DataFrame(flights)
    flight_file = passenger_dir / "flight_manifest.csv"
    df_flights.to_csv(flight_file, index=False)
    
    print(f"✓ Generated flight manifest with {len(flights)} flights")
    
    return passenger_dir

def main():
    """Main function to set up entire project"""
    
    print("=" * 60)
    print("🛫 ADS-B Anomaly Detection Project Setup")
    print("=" * 60)
    
    # 1. Setup folder structure
    project_root = setup_project_structure()
    
    # 2. Download OpenSky data
    raw_data_dir = download_opensky_datasets(project_root)
    
    # 3. Generate passenger data
    passenger_dir = generate_passenger_data(project_root)
    
    print("\n" + "=" * 60)
    print("✅ Project Setup Complete!")
    print("=" * 60)
    
    # Show summary
    print("\n📁 Project Structure Created:")
    print(f"   Root: {project_root}")
    print(f"   Raw Data: {raw_data_dir}")
    print(f"   Processed Data: {project_root / 'data' / 'opensky_processed'}")
    print(f"   Passenger Data: {passenger_dir}")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: python process_opensky_data.py (to convert JSON to CSV)")
    print("   2. Run: python app.py (to start the dashboard)")
    print("   3. Open: http://localhost:3000")

if __name__ == "__main__":
    main()