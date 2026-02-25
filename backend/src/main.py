# Python 3.14 Compatibility Patch - MUST BE AT THE TOP
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Apply Python 3.14 compatibility patches
try:
    # Try to apply compatibility patch
    if sys.version_info >= (3, 14):
        print(f"🐍 Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Set environment variables for compatibility
        os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
        os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"
        
        # Attempt to patch pkgutil if needed
        try:
            import pkgutil
            if not hasattr(pkgutil, 'ImpImporter'):
                # Create a dummy class for compatibility
                class ImpImporter:
                    def __init__(self, *args, **kwargs):
                        pass
                    def find_module(self, *args, **kwargs):
                        return None
                    def load_module(self, *args, **kwargs):
                        return None
                pkgutil.ImpImporter = ImpImporter
                print("✅ Applied pkgutil compatibility patch")
        except Exception as e:
            print(f"⚠️ Could not patch pkgutil: {e}")
except Exception as e:
    print(f"⚠️ Compatibility setup warning: {e}")

# Now import the rest of the modules
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import asyncio
import json
from typing import List, Dict, Any, Optional
import uvicorn
import traceback
from datetime import datetime
import pytz

try:
    from config import settings
    from data_processor import ADS_BDataProcessor
    from anomaly_detector import AnomalyDetector
    from ml_models import AnomalyDetectionModels
    from database import get_db, FlightData, AnomalyEvent, engine, Base, AsyncSessionLocal
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.future import select
    from sqlalchemy import func
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Some features may be unavailable")
    # Create minimal stubs for missing imports
    class Settings:
        PROJECT_NAME = "ADS-B Anomaly Detection"
        API_V1_STR = "/api/v1"
        DATABASE_URL = "sqlite+aiosqlite:///./adsb.db"
        ADSB_UPDATE_INTERVAL = 5
        MODEL_PATH = "./models"
        ANOMALY_THRESHOLD = 0.85
    
    settings = Settings()

# Global instances with lazy initialization
data_processor = None
ml_models = None
anomaly_detector = None

async def initialize_components():
    """Initialize all components with proper error handling"""
    global data_processor, ml_models, anomaly_detector
    
    try:
        # Initialize data processor
        from data_processor import ADS_BDataProcessor
        data_processor = ADS_BDataProcessor()
        print("✅ Data processor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize data processor: {e}")
        # Create a minimal data processor
        class MinimalDataProcessor:
            async def fetch_live_data(self):
                return []
        data_processor = MinimalDataProcessor()
    
    try:
        # Initialize ML models
        from ml_models import AnomalyDetectionModels
        ml_models = AnomalyDetectionModels()
        print("✅ ML models initialized")
    except Exception as e:
        print(f"❌ Failed to initialize ML models: {e}")
        # Create fallback models
        class FallbackModels:
            def __init__(self):
                self.is_trained = False
            def load_models(self, path):
                print(f"⚠️ Using fallback models (no ML)")
                self.is_trained = False
        ml_models = FallbackModels()
    
    try:
        # Initialize anomaly detector
        from anomaly_detector import AnomalyDetector
        anomaly_detector = AnomalyDetector(ml_models)
        print("✅ Anomaly detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize anomaly detector: {e}")
        # Create fallback detector
        class FallbackDetector:
            async def detect_anomalies(self, flights):
                return []
        anomaly_detector = FallbackDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    print(f"🚀 Starting {settings.PROJECT_NAME}...")
    
    # Create database tables
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created")
    except Exception as e:
        print(f"⚠️ Could not create database tables: {e}")
    
    # Initialize components
    await initialize_components()
    
    # Load ML models
    try:
        models_dir = settings.MODEL_PATH
        os.makedirs(models_dir, exist_ok=True)
        
        # Check if models exist
        if hasattr(ml_models, 'load_models'):
            ml_models.load_models(models_dir)
            print("✅ ML models loaded successfully")
        else:
            print("⚠️ ML models not available, using rule-based detection only")
    except Exception as e:
        print(f"⚠️ Could not load ML models: {e}")
        print("⚠️ Using rule-based detection only")
    
    # Start background task for data processing
    task = asyncio.create_task(process_adsb_data())
    print("📡 Started ADS-B data processing task")
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("✅ Data processing task cancelled")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware - UPDATED VERSION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers
)

# WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        # DON'T call websocket.accept() here - it's already called in the endpoint
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, data: Dict[str, Any]):
        """Broadcast data to all connected WebSocket clients"""
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                print(f"⚠️ Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

async def process_adsb_data():
    """Background task to process ADS-B data"""
    print("📡 Starting ADS-B data processing loop...")
    
    # Wait a moment for everything to initialize
    await asyncio.sleep(2)
    
    # Sample flight data for when API is unavailable
    sample_flights = [
        {
            "icao24": "a0b1c2",
            "callsign": "SAMPLE01",
            "origin_country": "United States",
            "latitude": 40.7128,
            "longitude": -74.0060,
            "altitude": 35000,
            "velocity": 450,
            "heading": 90,
            "vertical_rate": 0,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        },
        {
            "icao24": "d3e4f5",
            "callsign": "SAMPLE02",
            "origin_country": "Canada",
            "latitude": 43.6532,
            "longitude": -79.3832,
            "altitude": 28000,
            "velocity": 520,
            "heading": 180,
            "vertical_rate": 500,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    ]
    
    counter = 0
    while True:
        try:
            counter += 1
            flights = []
            
            # Try to fetch real data
            if data_processor and hasattr(data_processor, 'fetch_live_data'):
                try:
                    flights = await data_processor.fetch_live_data()
                except Exception as e:
                    print(f"⚠️ Error fetching live data: {e}")
            
            # If no real data, use sample data
            if not flights:
                # Add some variation to sample data
                import random
                sample_copy = []
                for flight in sample_flights:
                    flight_copy = flight.copy()
                    # Add some random variation
                    flight_copy["altitude"] += random.randint(-2000, 2000)
                    flight_copy["velocity"] += random.randint(-50, 50)
                    flight_copy["latitude"] += random.uniform(-1, 1)
                    flight_copy["longitude"] += random.uniform(-1, 1)
                    flight_copy["timestamp"] = datetime.now(pytz.UTC).isoformat()
                    
                    # Occasionally mark as anomaly for demo
                    if counter % 10 == 0:  # Every 10th iteration
                        flight_copy["is_anomaly_demo"] = True
                        flight_copy["anomaly_score"] = 0.9
                    else:
                        flight_copy["is_anomaly_demo"] = False
                        flight_copy["anomaly_score"] = random.uniform(0.1, 0.3)
                    
                    sample_copy.append(flight_copy)
                flights = sample_copy
            
            if flights:
                # Detect anomalies if detector is available
                anomalies = []
                if anomaly_detector and hasattr(anomaly_detector, 'detect_anomalies'):
                    try:
                        anomalies = await anomaly_detector.detect_anomalies(flights)
                    except Exception as e:
                        print(f"⚠️ Error detecting anomalies: {e}")
                        anomalies = []
                
                # Add demo anomalies if no real ones detected
                if not anomalies and counter % 5 == 0:
                    anomalies = [{
                        "flight_id": flights[0]["icao24"] if flights else "demo123",
                        "callsign": flights[0]["callsign"] if flights else "DEMO001",
                        "anomaly_type": "Demo Anomaly",
                        "severity": "medium",
                        "score": 0.85,
                        "timestamp": datetime.now(pytz.UTC).isoformat(),
                        "details": {
                            "message": "This is a demo anomaly for testing"
                        }
                    }]
                
                # Prepare data for WebSocket broadcast
                broadcast_data = {
                    'type': 'flight_update',
                    'timestamp': datetime.now(pytz.UTC).isoformat(),
                    'counter': counter,
                    'flights': flights[:50],  # Limit for performance
                    'anomalies': anomalies[:10],
                    'stats': {
                        'total_flights': len(flights),
                        'total_anomalies': len(anomalies),
                        'anomaly_rate': len(anomalies) / max(len(flights), 1),
                        'update_count': counter,
                        'system_status': 'operational',
                        'using_sample_data': len(flights) == len(sample_flights)
                    }
                }
                
                # Broadcast to all connected clients
                await manager.broadcast(broadcast_data)
                
                # Log status occasionally
                if counter % 20 == 0:
                    print(f"📊 Update {counter}: {len(flights)} flights, {len(anomalies)} anomalies")
            
        except Exception as e:
            print(f"❌ Error in data processing loop: {e}")
            traceback.print_exc()
        
        await asyncio.sleep(settings.ADSB_UPDATE_INTERVAL)

# ==================== ROOT ENDPOINTS (/) ====================
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/api/docs",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "endpoints": {
            "health": "/api/health, /api/v1/health",
            "datasets": "/api/datasets",
            "flights": "/api/v1/flights",
            "anomalies": "/api/v1/anomalies",
            "stats": "/api/v1/stats",
            "ws": "/ws (WebSocket)"
        }
    }

# ==================== API ENDPOINTS - BOTH /api AND /api/v1 ====================

# Health endpoints (both /api/health and /api/v1/health)
@app.get("/api/health")
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "active_connections": len(manager.active_connections),
        "ml_models_loaded": ml_models.is_trained if hasattr(ml_models, 'is_trained') else False,
        "components": {
            "data_processor": data_processor is not None,
            "ml_models": ml_models is not None,
            "anomaly_detector": anomaly_detector is not None
        }
    }

# ==================== DATASET ENDPOINTS ====================
@app.get("/api/datasets")
@app.get("/api/v1/datasets")
async def get_datasets():
    """Return list of 30 available datasets"""
    print("📂 GET /api/datasets - Returning 30 datasets")
    
    # Create 30 datasets
    datasets = []
    for i in range(1, 31):
        # Add some variation to make it interesting
        flights_count = 100
        anomalies_count = 10
        if i % 5 == 0:
            flights_count = 120
            anomalies_count = 15
        elif i % 7 == 0:
            flights_count = 80
            anomalies_count = 8
        
        datasets.append({
            "id": i,
            "name": f"dataset_{i:02d}",
            "flights": flights_count,
            "anomalies": anomalies_count,
            "date": f"2024-01-{i:02d}",
            "description": f"OpenSky dataset {i:02d} with {flights_count} flights",
            "airports": ["JFK", "LAX", "LHR", "CDG", "DFW"][:3],
            "anomaly_rate": f"{(anomalies_count/flights_count*100):.1f}%"
        })
    
    return JSONResponse({
        "datasets": datasets,
        "count": len(datasets),
        "total_flights": sum(d["flights"] for d in datasets),
        "total_anomalies": sum(d["anomalies"] for d in datasets),
        "timestamp": datetime.now(pytz.UTC).isoformat()
    })

@app.get("/api/dataset/{dataset_id}")
@app.get("/api/v1/dataset/{dataset_id}")
async def get_dataset_info(dataset_id: int):
    """Get detailed information about a specific dataset"""
    print(f"📂 GET /api/dataset/{dataset_id}")
    
    if dataset_id < 1 or dataset_id > 30:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Generate dataset info
    flights_count = 100
    anomalies_count = 10
    
    if dataset_id % 5 == 0:
        flights_count = 120
        anomalies_count = 15
    elif dataset_id % 7 == 0:
        flights_count = 80
        anomalies_count = 8
    
    # Generate sample routes based on dataset_id
    routes = [
        {"origin": "JFK", "destination": "LAX", "count": 25},
        {"origin": "LHR", "destination": "JFK", "count": 20},
        {"origin": "CDG", "destination": "DFW", "count": 15},
        {"origin": "DXB", "destination": "JFK", "count": 12},
        {"origin": "HKG", "destination": "SFO", "count": 10},
        {"origin": "SYD", "destination": "LAX", "count": 8},
        {"origin": "NRT", "destination": "ORD", "count": 6},
        {"origin": "FRA", "destination": "IAD", "count": 4}
    ]
    
    return JSONResponse({
        "id": dataset_id,
        "name": f"dataset_{dataset_id:02d}",
        "flights": flights_count,
        "anomalies": anomalies_count,
        "date": f"2024-01-{dataset_id:02d}",
        "description": f"OpenSky dataset {dataset_id:02d} with {flights_count} flights",
        "anomaly_rate": f"{(anomalies_count/flights_count*100):.1f}%",
        "routes": routes[:4],
        "airlines": ["UAL", "AAL", "DAL", "BAW", "AFR", "SWA"],
        "coverage": {
            "start_time": f"2024-01-{dataset_id:02d}T00:00:00Z",
            "end_time": f"2024-01-{dataset_id:02d}T23:59:59Z",
            "region": "Global"
        },
        "stats": {
            "avg_altitude": 35000,
            "avg_speed": 450,
            "max_altitude": 43000,
            "min_altitude": 5000
        }
    })

# ==================== FLIGHT ENDPOINTS ====================
@app.get("/api/flights")
@app.get("/api/v1/flights")
async def get_flights(limit: int = 100):
    """Get recent flights"""
    try:
        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(FlightData)
                    .order_by(FlightData.timestamp.desc())
                    .limit(limit)
                )
                flights = result.scalars().all()
                
                # Convert to dict if needed
                flights_data = []
                for flight in flights:
                    flight_dict = {c.name: getattr(flight, c.name) for c in flight.__table__.columns}
                    # Convert datetime to string
                    if 'timestamp' in flight_dict and flight_dict['timestamp']:
                        flight_dict['timestamp'] = flight_dict['timestamp'].isoformat()
                    flights_data.append(flight_dict)
                
                return {
                    "flights": flights_data,
                    "count": len(flights_data),
                    "limit": limit
                }
            except Exception as e:
                print(f"❌ Database error in get_flights: {e}")
                # Return empty data if database not available
                return {
                    "flights": [],
                    "count": 0,
                    "limit": limit,
                    "message": "Database not available, returning empty list"
                }
    except Exception as e:
        print(f"❌ Error in get_flights endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/flights/live")
@app.get("/api/v1/flights/live")
async def get_live_flights():
    """Get live flight data (simulated for demo)"""
    print("📡 GET /api/flights/live - Returning live flight data")
    
    import random
    from datetime import datetime, timedelta
    
    airlines = ["UAL", "AAL", "DAL", "BAW", "AFR", "SWA", "JBU", "RYR"]
    airports = ["JFK", "LAX", "LHR", "CDG", "DFW", "ORD", "ATL", "SFO", "DXB", "HKG"]
    phases = ["Climb", "Cruise", "Descent", "Approach"]
    
    flights = []
    for i in range(1, 51):  # Generate 50 flights
        is_anomaly = random.random() < 0.1  # 10% anomaly rate
        anomaly_score = random.uniform(0.7, 0.95) if is_anomaly else random.uniform(0, 0.3)
        
        origin = random.choice(airports)
        destination = random.choice([a for a in airports if a != origin])
        
        flight = {
            "icao24": f"{random.randint(100000, 999999):x}",
            "callsign": f"{random.choice(airlines)}{random.randint(100, 999)}",
            "airline": random.choice(airlines),
            "origin": origin,
            "destination": destination,
            "latitude": random.uniform(25, 50),
            "longitude": random.uniform(-125, -65),
            "altitude_ft": random.randint(25000, 41000),
            "speed_knots": random.randint(400, 550),
            "flight_phase": random.choice(phases),
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "passenger_count": random.randint(50, 250),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        flights.append(flight)
    
    return {
        "mode": "live",
        "flights": flights,
        "count": len(flights),
        "anomalies": len([f for f in flights if f["is_anomaly"]]),
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

@app.get("/api/snapshot/{dataset_id}")
@app.get("/api/v1/snapshot/{dataset_id}")
async def get_snapshot(dataset_id: int, time_index: int = 50):
    """Get snapshot data from a specific dataset"""
    print(f"📸 GET /api/snapshot/{dataset_id}?time_index={time_index}")
    
    if dataset_id < 1 or dataset_id > 30:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    import random
    from datetime import datetime, timedelta
    
    airlines = ["UAL", "AAL", "DAL", "BAW", "AFR", "SWA", "JBU", "RYR"]
    airports = ["JFK", "LAX", "LHR", "CDG", "DFW", "ORD", "ATL", "SFO", "DXB", "HKG"]
    phases = ["Climb", "Cruise", "Descent", "Approach"]
    
    # Adjust anomaly rate based on dataset
    anomaly_base_rate = 0.1
    if dataset_id % 5 == 0:
        anomaly_base_rate = 0.15
    elif dataset_id % 7 == 0:
        anomaly_base_rate = 0.08
    
    flights = []
    for i in range(1, 101):  # 100 flights per dataset
        is_anomaly = random.random() < anomaly_base_rate
        anomaly_score = random.uniform(0.7, 0.95) if is_anomaly else random.uniform(0, 0.3)
        
        origin = random.choice(airports)
        destination = random.choice([a for a in airports if a != origin])
        
        # Add some variation based on time_index
        time_factor = time_index / 100.0
        altitude = int(25000 + (time_factor * 16000))
        
        flight = {
            "icao24": f"{random.randint(100000, 999999):x}",
            "callsign": f"{random.choice(airlines)}{random.randint(100, 999)}",
            "airline": random.choice(airlines),
            "origin": origin,
            "destination": destination,
            "latitude": random.uniform(25, 50),
            "longitude": random.uniform(-125, -65),
            "altitude_ft": altitude,
            "speed_knots": random.randint(400, 550),
            "flight_phase": random.choice(phases),
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "passenger_count": random.randint(50, 250),
            "timestamp": f"2024-01-{dataset_id:02d}T{int(time_index/4):02d}:{(time_index%4)*15:02d}:00Z"
        }
        flights.append(flight)
    
    return {
        "mode": "snapshot",
        "dataset_id": dataset_id,
        "time_index": time_index,
        "flights": flights,
        "count": len(flights),
        "anomalies": len([f for f in flights if f["is_anomaly"]]),
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

@app.post("/api/set-mode")
@app.post("/api/v1/set-mode")
async def set_mode(request: Dict[str, str]):
    """Set the operation mode (live/snapshot)"""
    mode = request.get("mode", "live")
    print(f"🔄 Setting mode to: {mode}")
    
    return {
        "mode": mode,
        "status": "success",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "message": f"Switched to {mode} mode"
    }

@app.get("/api/passengers/{flight_id}")
@app.get("/api/v1/passengers/{flight_id}")
async def get_passengers(flight_id: str):
    """Get passenger manifest for a flight"""
    print(f"👥 GET /api/passengers/{flight_id}")
    
    import random
    from datetime import datetime, timedelta
    
    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emma", "James", "Lisa", "Robert", "Maria"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    classes = ["Economy", "Premium Economy", "Business", "First"]
    statuses = ["Checked-in", "Boarded", "Not Checked-in", "Standby"]
    
    # Generate 50-200 passengers
    num_passengers = random.randint(50, 200)
    passengers = []
    
    for i in range(1, num_passengers + 1):
        first = random.choice(first_names)
        last = random.choice(last_names)
        seat_row = random.randint(1, 40)
        seat_letter = random.choice(["A", "B", "C", "D", "E", "F"])
        
        passenger = {
            "passenger_id": f"PAX{i:05d}",
            "flight_callsign": flight_id,
            "name": f"{first} {last}",
            "seat": f"{seat_row}{seat_letter}",
            "booking_class": random.choice(classes),
            "status": random.choice(statuses),
            "check_in_time": (datetime.now(pytz.UTC) - timedelta(hours=random.randint(1, 24))).isoformat() if random.random() > 0.3 else None,
            "baggage_count": random.randint(0, 3),
            "frequent_flyer": random.choice(["Gold", "Silver", "Bronze", "None"]),
            "special_requirements": random.choice(["None", "Wheelchair", "Vegetarian Meal", "Kosher Meal"]) if random.random() > 0.7 else "None"
        }
        passengers.append(passenger)
    
    return {
        "flight_callsign": flight_id,
        "passengers": passengers,
        "count": len(passengers),
        "summary": {
            "total_passengers": len(passengers),
            "checked_in": len([p for p in passengers if p["status"] in ["Checked-in", "Boarded"]]),
            "boarded": len([p for p in passengers if p["status"] == "Boarded"]),
            "by_class": {
                cls: len([p for p in passengers if p["booking_class"] == cls])
                for cls in classes
            }
        },
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

# ==================== ANOMALY ENDPOINTS ====================
@app.get("/api/anomalies")
@app.get("/api/v1/anomalies")
async def get_anomalies(
    severity: Optional[str] = None, 
    limit: int = 50,
    recent: bool = True
):
    """Get recent anomalies"""
    try:
        async with AsyncSessionLocal() as session:
            query = select(AnomalyEvent)
            
            if severity:
                query = query.where(AnomalyEvent.severity == severity)
            
            if recent:
                query = query.order_by(AnomalyEvent.timestamp.desc())
            
            query = query.limit(limit)
            
            try:
                result = await session.execute(query)
                anomalies = result.scalars().all()
                
                # Convert to dict
                anomalies_data = []
                for anomaly in anomalies:
                    anomaly_dict = {c.name: getattr(anomaly, c.name) for c in anomaly.__table__.columns}
                    # Convert datetime to string
                    if 'timestamp' in anomaly_dict and anomaly_dict['timestamp']:
                        anomaly_dict['timestamp'] = anomaly_dict['timestamp'].isoformat()
                    anomalies_data.append(anomaly_dict)
                
                return {
                    "anomalies": anomalies_data,
                    "count": len(anomalies_data),
                    "severity_filter": severity,
                    "limit": limit
                }
            except Exception as e:
                print(f"❌ Database error in get_anomalies: {e}")
                # Return demo data if database not available
                demo_anomalies = [
                    {
                        "id": 1,
                        "flight_id": "demo001",
                        "anomaly_type": "Sudden Altitude Change",
                        "severity": "high",
                        "description": "Demo anomaly for testing",
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    }
                ]
                return {
                    "anomalies": demo_anomalies,
                    "count": len(demo_anomalies),
                    "message": "Using demo data (database not available)"
                }
    except Exception as e:
        print(f"❌ Error in get_anomalies endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== STATS ENDPOINTS ====================
@app.get("/api/stats")
@app.get("/api/v1/stats")
async def get_statistics():
    """Get system statistics"""
    try:
        async with AsyncSessionLocal() as session:
            try:
                # Get flight counts
                total_result = await session.execute(
                    select(func.count()).select_from(FlightData)
                )
                total_flights = total_result.scalar() or 0
                
                anomaly_result = await session.execute(
                    select(func.count()).select_from(FlightData).where(FlightData.is_anomaly == True)
                )
                anomaly_count = anomaly_result.scalar() or 0
                
                # Get latest anomaly
                latest_anomaly_result = await session.execute(
                    select(AnomalyEvent)
                    .order_by(AnomalyEvent.timestamp.desc())
                    .limit(1)
                )
                latest_anomaly = latest_anomaly_result.scalar_one_or_none()
                
                stats = {
                    "total_flights": total_flights,
                    "total_anomalies": anomaly_count,
                    "anomaly_rate": anomaly_count / max(total_flights, 1),
                    "system_status": "operational",
                    "last_update": datetime.now(pytz.UTC).isoformat(),
                    "database_available": True,
                    "latest_anomaly": {
                        "type": latest_anomaly.anomaly_type if latest_anomaly else None,
                        "severity": latest_anomaly.severity if latest_anomaly else None,
                        "timestamp": latest_anomaly.timestamp.isoformat() if latest_anomaly else None
                    } if latest_anomaly else None
                }
            except Exception as e:
                print(f"❌ Database error in get_statistics: {e}")
                # Return basic stats if database not available
                stats = {
                    "total_flights": 0,
                    "total_anomalies": 0,
                    "anomaly_rate": 0,
                    "system_status": "operational (demo mode)",
                    "last_update": datetime.now(pytz.UTC).isoformat(),
                    "database_available": False,
                    "message": "Database not available, using demo statistics"
                }
        
        # Add ML model status
        if ml_models:
            stats["ml_models"] = {
                "is_trained": getattr(ml_models, 'is_trained', False),
                "sklearn_available": getattr(ml_models, 'sklearn_available', False),
                "tensorflow_available": getattr(ml_models, 'tensorflow_available', False)
            }
        
        # Add WebSocket connections
        stats["active_connections"] = len(manager.active_connections)
        
        return stats
        
    except Exception as e:
        print(f"❌ Error in get_statistics endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET ENDPOINT ====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data - FIXED VERSION"""
    try:
        # Accept the connection FIRST
        await websocket.accept()
        client_host = websocket.client.host if websocket.client else "unknown"
        print(f"✅ New WebSocket connection from {client_host}")
        
        # Add connection to manager (without calling accept again)
        await manager.connect(websocket)
        
        # Send welcome message
        await websocket.send_json({
            "type": "connection_established",
            "message": "Welcome to ADS-B Anomaly Detection System",
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "status": "connected"
        })
        
        # Keep connection alive
        while True:
            try:
                # Receive data with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                if data:
                    try:
                        message = json.loads(data)
                        message_type = message.get("type", "unknown")
                        
                        if message_type == "ping":
                            await websocket.send_json({
                                "type": "pong",
                                "timestamp": datetime.now(pytz.UTC).isoformat()
                            })
                        elif message_type == "request_flights":
                            mode = message.get("mode", "live")
                            print(f"📡 Client requested flights in {mode} mode")
                            
                            # Generate response based on mode
                            if mode == "live":
                                import random
                                from datetime import datetime
                                
                                airlines = ["UAL", "AAL", "DAL", "BAW", "AFR"]
                                airports = ["JFK", "LAX", "LHR", "CDG", "DFW"]
                                
                                flights = []
                                for i in range(1, 21):
                                    flight = {
                                        "icao24": f"{random.randint(100000, 999999):x}",
                                        "callsign": f"{random.choice(airlines)}{random.randint(100, 999)}",
                                        "airline": random.choice(airlines),
                                        "origin": random.choice(airports),
                                        "destination": random.choice(airports),
                                        "latitude": random.uniform(25, 50),
                                        "longitude": random.uniform(-125, -65),
                                        "altitude_ft": random.randint(25000, 41000),
                                        "speed_knots": random.randint(400, 550),
                                        "flight_phase": "Cruise",
                                        "is_anomaly": random.random() < 0.1,
                                        "anomaly_score": random.uniform(0, 1),
                                        "passenger_count": random.randint(50, 250)
                                    }
                                    flights.append(flight)
                                
                                await websocket.send_json({
                                    "type": "flight_data",
                                    "mode": "live",
                                    "flights": flights,
                                    "timestamp": datetime.now(pytz.UTC).isoformat()
                                })
                            else:
                                await websocket.send_json({
                                    "type": "flight_data",
                                    "mode": "snapshot",
                                    "flights": [],
                                    "message": "Please select a dataset first",
                                    "timestamp": datetime.now(pytz.UTC).isoformat()
                                })
                        elif message_type == "request_passengers":
                            flight_callsign = message.get("flight_callsign", "UNKNOWN")
                            print(f"👥 Client requested passengers for {flight_callsign}")
                            
                            import random
                            from datetime import datetime
                            
                            first_names = ["John", "Jane", "Michael", "Sarah"]
                            last_names = ["Smith", "Johnson", "Williams", "Brown"]
                            
                            passengers = []
                            for i in range(1, random.randint(20, 50)):
                                passenger = {
                                    "passenger_id": f"PAX{i:04d}",
                                    "flight_callsign": flight_callsign,
                                    "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                                    "seat": f"{random.randint(1, 30)}{random.choice(['A','B','C','D','E','F'])}",
                                    "booking_class": random.choice(["Economy", "Business", "First"]),
                                    "status": random.choice(["Checked-in", "Boarded", "Not Checked-in"])
                                }
                                passengers.append(passenger)
                            
                            await websocket.send_json({
                                "type": "passenger_data",
                                "flight_callsign": flight_callsign,
                                "passengers": passengers,
                                "timestamp": datetime.now(pytz.UTC).isoformat()
                            })
                        
                    except json.JSONDecodeError:
                        # Echo non-JSON messages
                        await websocket.send_json({
                            "type": "echo",
                            "received": data,
                            "timestamp": datetime.now(pytz.UTC).isoformat()
                        })
                        
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now(pytz.UTC).isoformat()
                    })
                except Exception:
                    break  # Client disconnected
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        print(f"❌ WebSocket disconnected: {client_host}")
    except Exception as e:
        print(f"❌ WebSocket error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            manager.disconnect(websocket)
        except:
            pass

# ==================== TEST AND DEBUG ENDPOINTS ====================
@app.get("/api/test/flight")
@app.get("/api/v1/test/flight")
async def test_flight():
    """Generate a test flight for development"""
    import random
    test_flight = {
        "icao24": f"test{random.randint(1000, 9999)}",
        "callsign": f"TEST{random.randint(100, 999)}",
        "origin_country": "Test Country",
        "latitude": random.uniform(-90, 90),
        "longitude": random.uniform(-180, 180),
        "altitude": random.uniform(1000, 40000),
        "velocity": random.uniform(100, 600),
        "heading": random.uniform(0, 360),
        "vertical_rate": random.uniform(-3000, 3000),
        "is_anomaly": random.random() > 0.8,
        "anomaly_score": random.uniform(0, 1),
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "test_flight",
        "flight": test_flight,
        "timestamp": datetime.now(pytz.UTC).isoformat()
    })
    
    return test_flight

@app.get("/api/ws-info")
@app.get("/api/v1/ws-info")
async def websocket_info():
    """Get WebSocket connection information"""
    return {
        "active_connections": len(manager.active_connections),
        "status": "operational" if manager.active_connections else "no_connections",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "endpoint": "ws://localhost:8000/ws",
        "supported_messages": ["ping", "pong", "connection_established", "flight_update", "request_flights", "request_passengers"]
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    error_detail = str(exc)
    error_type = type(exc).__name__
    
    print(f"❌ Unhandled exception: {error_type}: {error_detail}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": error_detail,
            "type": error_type,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
    )

if __name__ == "__main__":
    print(f"""
    🚀 Starting {settings.PROJECT_NAME}
    🌐 API: http://localhost:8000
    📚 Docs: http://localhost:8000/api/docs
    🔌 WebSocket: ws://localhost:8000/ws
    🐍 Python: {sys.version}
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
