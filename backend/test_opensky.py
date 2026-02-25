#!/usr/bin/env python3
"""
Test OpenSky Network connection
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_processor import ADS_BDataProcessor

async def test():
    print("🔍 Testing OpenSky Network connection...")
    
    try:
        processor = ADS_BDataProcessor()
        flights = await processor.fetch_live_data()
        
        if flights:
            print(f"✅ SUCCESS: Received {len(flights)} flights")
            print("\n📊 Sample flight data:")
            for i, flight in enumerate(flights[:3]):  # Show first 3
                print(f"\nFlight {i+1}:")
                print(f"  Callsign: {flight.get('callsign')}")
                print(f"  Country: {flight.get('origin_country')}")
                print(f"  Position: {flight.get('latitude'):.2f}, {flight.get('longitude'):.2f}")
                print(f"  Altitude: {flight.get('altitude')} ft")
                print(f"  Speed: {flight.get('velocity')} kts")
            
            print(f"\n📈 Statistics:")
            print(f"  Total flights: {len(flights)}")
            
            # Count by country
            countries = {}
            for flight in flights:
                country = flight.get('origin_country', 'Unknown')
                countries[country] = countries.get(country, 0) + 1
            
            print(f"  Countries: {len(countries)}")
            for country, count in list(countries.items())[:5]:
                print(f"    {country}: {count} flights")
                
            return True
        else:
            print("❌ No flight data received")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test())
    
    if result:
        print("\n🎉 OpenSky Network is working!")
        print("\n💡 Next steps:")
        print("1. Restart your backend")
        print("2. Check dashboard for real flight data")
        print("3. Monitor backend logs for 'Fetched X real flights'")
    else:
        print("\n⚠️ OpenSky Network not accessible")
        print("\n🔧 Possible issues:")
        print("1. No internet connection")
        print("2. OpenSky API might be down")
        print("3. Firewall blocking connection")
        print("4. Rate limit exceeded (wait 60 seconds)")
        print("\n💡 Your system will use simulated data as fallback")