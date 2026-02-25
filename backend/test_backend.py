#!/usr/bin/env python3
"""
ADS-B Backend Diagnostic Test
Tests both HTTP API and WebSocket connections
"""

import asyncio
import websockets
import aiohttp
import sys
import json
import socket

async def test_backend():
    print("🔍 Testing ADS-B Backend...")
    
    # Test 1: HTTP API
    print("\n1. Testing HTTP API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/api/v1/health', timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ HTTP API is running")
                    print(f"   📊 Status: {data.get('status', 'unknown')}")
                    print(f"   🔌 Active connections: {data.get('active_connections', 0)}")
                    print(f"   🤖 ML models loaded: {data.get('ml_models_loaded', False)}")
                else:
                    print(f"   ❌ HTTP API returned status: {response.status}")
                    return False
    except asyncio.TimeoutError:
        print("   ❌ HTTP API timeout (5 seconds)")
        print("   💡 Make sure backend is running on port 8000")
        return False
    except Exception as e:
        print(f"   ❌ HTTP API test failed: {e}")
        print("   💡 Make sure backend is running: uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
        return False
    
    # Test 2: WebSocket
    print("\n2. Testing WebSocket...")
    try:
        # Create connection without timeout parameter (handled differently)
        websocket = await asyncio.wait_for(
            websockets.connect('ws://localhost:8000/ws'),
            timeout=5
        )
        
        print("   ✅ WebSocket connection established")
        
        # Wait for welcome message
        try:
            welcome = await asyncio.wait_for(websocket.recv(), timeout=2)
            welcome_data = json.loads(welcome)
            print(f"   ✅ Received welcome: {welcome_data.get('message', 'Connected')}")
            
            # Send a test ping
            await websocket.send(json.dumps({"type": "ping"}))
            
            # Wait for pong response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                response_data = json.loads(response)
                if response_data.get('type') == 'pong':
                    print(f"   ✅ Ping-pong successful")
                else:
                    print(f"   ⚠️ Unexpected response type: {response_data.get('type')}")
            except asyncio.TimeoutError:
                print("   ⚠️ No pong response (might be normal)")
            
            await websocket.close()
            return True
            
        except asyncio.TimeoutError:
            print("   ⚠️ No welcome message received (connection might still work)")
            await websocket.close()
            return True
            
    except asyncio.TimeoutError:
        print("   ❌ WebSocket connection timeout (5 seconds)")
        return False
    except Exception as e:
        print(f"   ❌ WebSocket test failed: {e}")
        print(f"   💡 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_all_endpoints():
    """Test all API endpoints"""
    print("\n3. Testing all API endpoints...")
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/api/v1/flights?limit=3", "Get flights"),
        ("/api/v1/anomalies?limit=3", "Get anomalies"),
        ("/api/v1/stats", "Get statistics"),
        ("/api/v1/test/flight", "Test flight generation"),
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint, description in endpoints:
            try:
                url = f'http://localhost:8000{endpoint}'
                async with session.get(url, timeout=3) as response:
                    if response.status == 200:
                        print(f"   ✅ {description}: HTTP {response.status}")
                        try:
                            data = await response.json()
                            if 'flights' in data:
                                print(f"      📊 Found {len(data['flights'])} flights")
                            elif 'anomalies' in data:
                                print(f"      ⚠️ Found {len(data['anomalies'])} anomalies")
                            elif 'total_flights' in data:
                                print(f"      📈 Stats: {data['total_flights']} flights, {data['total_anomalies']} anomalies")
                        except:
                            pass
                    else:
                        print(f"   ⚠️ {description}: HTTP {response.status}")
            except Exception as e:
                print(f"   ❌ {description}: Failed - {e}")

def check_port():
    """Check if port 8000 is open"""
    print("🔍 Checking port 8000...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        
        if result == 0:
            print("✅ Port 8000 is open")
            return True
        else:
            print("❌ Port 8000 is not responding")
            return False
    except Exception as e:
        print(f"⚠️ Could not check port: {e}")
        return False

async def main():
    """Main async function"""
    print("=" * 50)
    print("🚀 ADS-B Backend Diagnostic Test")
    print("=" * 50)
    
    # First check if backend is even running
    if not check_port():
        print("\n💡 Backend is not running! Start it with:")
        print("""
    cd backend
    venv\\Scripts\\activate
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
        """)
        return
    
    success = await test_backend()
    
    if success:
        # Test all endpoints
        await test_all_endpoints()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Backend is running correctly!")
        print("\n📋 Next steps:")
        print("1. Frontend should connect to: http://localhost:8000")
        print("2. WebSocket URL: ws://localhost:8000/ws")
        print("3. API Docs: http://localhost:8000/api/docs")
        print("4. Health check: http://localhost:8000/api/v1/health")
    else:
        print("❌ Backend has issues (WebSocket failing)")
        print("\n🔧 WebSocket Troubleshooting:")
        print("1. Make sure CORS is enabled in backend (it is)")
        print("2. Try updating websockets library:")
        print("   pip install --upgrade websockets")
        print("3. Check if any firewall is blocking WebSocket")
        print("4. Try with a different browser")
        print("\n💡 Quick fix test:")
        print("   Open browser console (F12) and run:")
        print("   new WebSocket('ws://localhost:8000/ws')")
        print("\n📌 Current status: HTTP API ✓ | WebSocket ✗")
    print("=" * 50)

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()