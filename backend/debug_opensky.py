#!/usr/bin/env python3
"""
Debug OpenSky Network connection issues
"""

import asyncio
import aiohttp
import json
import sys
import socket

async def debug_opensky():
    print("🔍 Debugging OpenSky Network connection...")
    
    # Test 1: Basic internet connectivity
    print("\n1. Testing internet connectivity...")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print("✅ Internet connection OK")
    except:
        print("❌ No internet connection")
        return False
    
    # Test 2: DNS resolution
    print("\n2. Testing DNS resolution...")
    try:
        ip = socket.gethostbyname("opensky-network.org")
        print(f"✅ DNS resolved: opensky-network.org → {ip}")
    except:
        print("❌ DNS resolution failed")
        return False
    
    # Test 3: Direct HTTP request
    print("\n3. Testing direct HTTP request...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://opensky-network.org", timeout=10) as response:
                print(f"✅ Website accessible: HTTP {response.status}")
    except Exception as e:
        print(f"❌ Website not accessible: {e}")
    
    # Test 4: API endpoint test
    print("\n4. Testing API endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://opensky-network.org/api/states/all",
                timeout=10
            ) as response:
                print(f"✅ API endpoint: HTTP {response.status}")
                
                if response.status == 200:
                    data = await response.text()
                    print(f"✅ Received {len(data)} bytes")
                    
                    # Try to parse JSON
                    try:
                        json_data = json.loads(data)
                        if 'states' in json_data:
                            states = json_data['states']
                            print(f"✅ Parsed {len(states) if states else 0} flight states")
                            if states and len(states) > 0:
                                print(f"✅ First flight: {states[0][1] if states[0][1] else 'Unknown'}")
                                return True
                        else:
                            print("⚠️ No 'states' in response")
                            print(f"Response keys: {list(json_data.keys())}")
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON parse error: {e}")
                        print(f"First 200 chars: {data[:200]}")
                elif response.status == 429:
                    print("❌ Rate limited - wait 60 seconds")
                elif response.status == 403:
                    print("❌ Forbidden - IP might be blocked")
                else:
                    print(f"❌ Unexpected status: {response.status}")
                    
    except Exception as e:
        print(f"❌ API request failed: {e}")
        import traceback
        traceback.print_exc()
    
    return False

async def test_alternative_endpoints():
    """Test alternative OpenSky endpoints"""
    print("\n🔧 Testing alternative endpoints...")
    
    endpoints = [
        ("https://opensky-network.org/api/states/all?time=0", "Global all states"),
        ("https://opensky-network.org/api/states/all?lamin=45.8389&lomin=5.9962&lamax=47.8229&lomax=10.5226", "Europe region"),
        ("https://opensky-network.org/api/states/all?lamin=20&lomin=-130&lamax=55&lomax=-60", "North America region"),
    ]
    
    for url, description in endpoints:
        print(f"\n  Testing {description}...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'states' in data and data['states']:
                            print(f"    ✅ {len(data['states'])} flights")
                        else:
                            print(f"    ⚠️ No flights")
                    else:
                        print(f"    ❌ HTTP {response.status}")
        except Exception as e:
            print(f"    ❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🛰️ OpenSky Network Debug Tool")
    print("=" * 60)
    
    success = asyncio.run(debug_opensky())
    
    if not success:
        print("\n" + "=" * 60)
        print("🔄 Trying alternative endpoints...")
        print("=" * 60)
        asyncio.run(test_alternative_endpoints())
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 OpenSky Network is working!")
        print("\n💡 Update your backend to use real data")
    else:
        print("⚠️ OpenSky Network has issues")
        print("\n🔧 Next steps:")
        print("1. Try again in 5 minutes (might be temporary)")
        print("2. Check firewall/proxy settings")
        print("3. Use alternative data source (see below)")
        print("4. Continue with simulated data for now")
    print("=" * 60)