#!/usr/bin/env python3
"""
Test all available ADS-B data sources
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_processor import ADS_BDataProcessor

async def test_all_sources():
    print("🔍 Testing all ADS-B data sources...")
    print("=" * 60)
    
    processor = ADS_BDataProcessor()
    
    # Test each source individually
    print("\n1. Testing OpenSky Network...")
    try:
        opensky_flights = await processor._fetch_opensky()
        print(f"   Result: {len(opensky_flights)} flights")
        if opensky_flights:
            print(f"   Sample: {opensky_flights[0].get('callsign')} at {opensky_flights[0].get('latitude'):.1f}, {opensky_flights[0].get('longitude'):.1f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Testing simulated data...")
    try:
        simulated_flights = processor._generate_realistic_simulation()
        print(f"   Result: {len(simulated_flights)} flights")
        if simulated_flights:
            print(f"   Sample: {simulated_flights[0].get('callsign')} on route {simulated_flights[0].get('route')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Testing main fetch method (automatic source selection)...")
    try:
        all_flights = await processor.fetch_live_data()
        print(f"   Result: {len(all_flights)} flights")
        if all_flights:
            source = all_flights[0].get('data_source', 'unknown')
            print(f"   Source selected: {source}")
            print(f"   Sample flight: {all_flights[0].get('callsign')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    # Final test
    flights = await processor.fetch_live_data()
    if flights:
        source = flights[0].get('data_source', 'unknown')
        print(f"✅ System is working with {source} data")
        print(f"📡 {len(flights)} flights available")
        
        # Show sample flights
        print("\n✈️ Sample flights:")
        for i, flight in enumerate(flights[:5]):
            print(f"  {i+1}. {flight.get('callsign'):8} | "
                  f"{flight.get('origin_country'):15} | "
                  f"Pos: {flight.get('latitude'):6.1f}, {flight.get('longitude'):7.1f} | "
                  f"Alt: {flight.get('altitude'):6} ft | "
                  f"Spd: {flight.get('velocity'):3} kts")
    else:
        print("❌ No flight data available from any source")

if __name__ == "__main__":
    asyncio.run(test_all_sources())
    
    print("\n💡 Next steps:")
    print("1. If OpenSky works, you'll see real flight data")
    print("2. If not, simulated data will provide realistic flights")
    print("3. Restart backend to use the updated data processor")