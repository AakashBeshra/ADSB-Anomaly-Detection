#!/usr/bin/env python3
"""
Simple WebSocket test without complex dependencies
"""

import websocket
import json
import time

def test_websocket():
    print("🔌 Testing WebSocket connection...")
    
    def on_message(ws, message):
        print(f"📨 Received: {message[:100]}...")
        data = json.loads(message)
        print(f"   Type: {data.get('type')}")
        if data.get('type') == 'connection_established':
            print("   ✅ Connection established message received!")
            # Send a ping
            ws.send(json.dumps({"type": "ping"}))
    
    def on_error(ws, error):
        print(f"❌ WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        print("✅ WebSocket connection opened!")
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        "ws://localhost:8000/ws",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    print("⏳ Connecting to WebSocket...")
    # Run for 5 seconds then close
    ws.run_forever()
    time.sleep(5)
    ws.close()

if __name__ == "__main__":
    try:
        test_websocket()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()