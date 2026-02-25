# backend/app.py (updated)
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from dataset_manager import DatasetManager
import json

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize dataset manager
dataset_manager = DatasetManager()

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')

@app.route('/snapshot')
def snapshot():
    """Serve snapshot dashboard"""
    return render_template('snapshot.html')

# API Routes
@app.route('/api/datasets')
def get_datasets():
    """Get list of all datasets"""
    datasets = dataset_manager.get_dataset_list()
    return jsonify(datasets)

@app.route('/api/dataset/<dataset_id>')
def get_dataset(dataset_id):
    """Get specific dataset info"""
    info = dataset_manager.get_dataset_info(dataset_id)
    return jsonify(info)

@app.route('/api/snapshot/<dataset_id>')
def get_dataset_snapshot(dataset_id):
    """Get snapshot from specific dataset"""
    time_index = request.args.get('time_index', 0, type=int)
    num_flights = request.args.get('num_flights', 50, type=int)
    
    flights = dataset_manager.get_snapshot(dataset_id, time_index, num_flights)
    return jsonify(flights)

@app.route('/api/live-flights')
def get_live_flights():
    """Get live flight data"""
    flights = dataset_manager.get_live_data()
    return jsonify(flights)

@app.route('/api/passengers/<flight_callsign>')
def get_passengers(flight_callsign):
    """Get passengers for a specific flight"""
    passengers = dataset_manager._generate_passengers_for_flight(flight_callsign)
    return jsonify(passengers)

@app.route('/api/set-mode', methods=['POST'])
def set_mode():
    """Set display mode"""
    data = request.json
    mode = data.get('mode', 'live')
    dataset_id = data.get('dataset_id')
    time_index = data.get('time_index', 0)
    
    dataset_manager.set_mode(mode)
    if dataset_id:
        dataset_manager.current_dataset = dataset_id
    if time_index:
        dataset_manager.current_time_index = time_index
    
    return jsonify({'status': 'success', 'mode': mode})

# WebSocket for live updates
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to ADS-B server'})

@socketio.on('request_flights')
def handle_request_flights(data):
    """Handle flight data requests"""
    mode = data.get('mode', 'live')
    
    if mode == 'live':
        flights = dataset_manager.get_live_data()
        emit('flight_data', {'flights': flights, 'mode': 'live'})
    elif mode == 'snapshot':
        dataset_id = data.get('dataset_id')
        time_index = data.get('time_index', 0)
        if dataset_id:
            flights = dataset_manager.get_snapshot(dataset_id, time_index)
            emit('flight_data', {'flights': flights, 'mode': 'snapshot', 'dataset_id': dataset_id})

@socketio.on('request_passengers')
def handle_request_passengers(data):
    """Handle passenger data requests"""
    flight_callsign = data.get('flight_callsign')
    if flight_callsign:
        passengers = dataset_manager._generate_passengers_for_flight(flight_callsign)
        emit('passenger_data', {'flight_callsign': flight_callsign, 'passengers': passengers})

if __name__ == '__main__':
    print("🚀 Starting ADS-B Anomaly Detection Server...")
    print(f"📁 Using {len(dataset_manager.datasets)} datasets")
    print(f"🌐 Live mode: Available")
    print(f"📸 Snapshot mode: Available with 30 datasets")
    print(f"👥 Passenger data: Integrated")
    socketio.run(app, debug=True, port=3000, allow_unsafe_werkzeug=True)
