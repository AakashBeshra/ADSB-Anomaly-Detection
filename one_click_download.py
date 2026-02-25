# one_click_download.py
"""
One-click script to download and process 30 OpenSky datasets
Run: python one_click_download.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        'boto3',
        'pandas',
        'numpy'
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✓ Installed {package}")
        except:
            print(f"  ✗ Failed to install {package}")
            return False
    
    return True

def run_downloader():
    """Run the main downloader"""
    print("\n🚀 Starting OpenSky data download...")
    
    # Import and run the downloader
    try:
        from opensky_s3_downloader import OpenSkyS3Downloader
        
        downloader = OpenSkyS3Downloader(output_dir="data/opensky")
        metadata = downloader.run_full_pipeline(num_datasets=30)
        
        return metadata
        
    except ImportError as e:
        print(f"Error importing downloader: {e}")
        return None

def create_project_structure():
    """Create project directory structure"""
    directories = [
        'data/opensky/raw',
        'data/opensky/processed',
        'data/passenger',
        'backend',
        'frontend',
        'ml_model'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created {directory}/")

def main():
    """Main function"""
    print("=" * 60)
    print("📡 OpenSky Data Downloader")
    print("=" * 60)
    
    # Create project structure
    print("\n📁 Creating project structure...")
    create_project_structure()
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements!")
        return
    
    # Run downloader
    metadata = run_downloader()
    
    if metadata:
        print("\n" + "=" * 60)
        print("🎉 Setup Complete!")
        print("=" * 60)
        print("\nYour 30 OpenSky datasets are ready!")
        print("\n📂 Files created:")
        print("  data/opensky/raw/           - Raw JSON.gz files")
        print("  data/opensky/processed/     - Processed CSV files")
        print("  data/opensky/processed/metadata.json - Dataset information")
        print("\n🚀 Next steps:")
        print("  1. Check the data: python check_data.py")
        print("  2. Update your dashboard to use the new datasets")
        print("  3. Run: python app.py to start your dashboard")
        
        # Create a simple check script
        create_check_script()

def create_check_script():
    """Create a script to check the downloaded data"""
    check_script = """# check_data.py
import pandas as pd
import json
import os

print("🔍 Checking downloaded OpenSky data...")

# Check processed directory
processed_dir = "data/opensky/processed"
if os.path.exists(processed_dir):
    print(f"✓ Found processed directory: {processed_dir}")
    
    # List CSV files
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    print(f"✓ Found {len(csv_files)} CSV datasets")
    
    if csv_files:
        # Check first dataset
        first_file = os.path.join(processed_dir, sorted(csv_files)[0])
        df = pd.read_csv(first_file)
        
        print(f"\\n📊 First dataset: {os.path.basename(first_file)}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Sample flight: {df['callsign'].iloc[0] if not df.empty else 'N/A'}")
        print(f"   Anomalies: {df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 'N/A'}")
    
    # Check metadata
    metadata_file = os.path.join(processed_dir, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\\n📋 Metadata:")
        print(f"   Total datasets: {metadata.get('total_datasets', 0)}")
        print(f"   Total flights: {metadata.get('total_flights', 0):,}")
else:
    print(f"✗ Processed directory not found: {processed_dir}")

print("\\n✅ Check complete!")"""
    
    with open("check_data.py", "w") as f:
        f.write(check_script)
    
    print("  ✓ Created check_data.py")

if __name__ == "__main__":
    main()