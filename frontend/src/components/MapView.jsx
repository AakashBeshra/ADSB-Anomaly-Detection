import React, { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet'
import L from 'leaflet'
import { Box, Typography, Chip } from '@mui/material'
import { Warning as WarningIcon } from '@mui/icons-material'

// Custom icons
const normalFlightIcon = new L.Icon({
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
})

const anomalyIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
})

const MapView = ({ flights, anomalies }) => {
  const mapRef = useRef(null)
  
  // Extract anomaly flight IDs
  const anomalyFlightIds = new Set(anomalies.map(a => a.flight_id))
  
  // Filter flights with valid coordinates
  const validFlights = flights.filter(f => 
    f.latitude && f.longitude && 
    Math.abs(f.latitude) <= 90 && 
    Math.abs(f.longitude) <= 180
  )

  // Center map on first flight or default to world center
  const center = validFlights.length > 0 
    ? [validFlights[0].latitude, validFlights[0].longitude]
    : [20, 0]

  useEffect(() => {
    if (mapRef.current && validFlights.length > 0) {
      const map = mapRef.current
      
      // Create bounds to fit all flights
      const bounds = L.latLngBounds(validFlights.map(f => [f.latitude, f.longitude]))
      map.fitBounds(bounds, { padding: [50, 50] })
    }
  }, [validFlights])

  return (
    <Box sx={{ height: '100%', width: '100%' }}>
      <MapContainer
        center={center}
        zoom={3}
        style={{ height: '100%', width: '100%', borderRadius: '8px' }}
        ref={mapRef}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {validFlights.map((flight) => {
          const isAnomaly = anomalyFlightIds.has(flight.icao24)
          const anomalyData = anomalies.find(a => a.flight_id === flight.icao24)
          
          return (
            <React.Fragment key={flight.icao24}>
              <Marker
                position={[flight.latitude, flight.longitude]}
                icon={isAnomaly ? anomalyIcon : normalFlightIcon}
              >
                <Popup>
                  <Box sx={{ minWidth: 200 }}>
                    <Typography variant="h6" gutterBottom>
                      {flight.callsign || 'UNKNOWN'}
                    </Typography>
                    
                    {isAnomaly && (
                      <Chip
                        icon={<WarningIcon />}
                        label="ANOMALY DETECTED"
                        color="error"
                        size="small"
                        sx={{ mb: 1 }}
                      />
                    )}
                    
                    <Typography variant="body2">
                      <strong>ICAO24:</strong> {flight.icao24}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Country:</strong> {flight.origin_country}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Altitude:</strong> {Math.round(flight.altitude || 0)} ft
                    </Typography>
                    <Typography variant="body2">
                      <strong>Speed:</strong> {Math.round(flight.velocity || 0)} kts
                    </Typography>
                    <Typography variant="body2">
                      <strong>Heading:</strong> {Math.round(flight.heading || 0)}°
                    </Typography>
                    
                    {anomalyData && (
                      <Box sx={{ mt: 1, p: 1, bgcolor: 'error.light', borderRadius: 1 }}>
                        <Typography variant="body2" color="white">
                          <strong>Anomaly:</strong> {anomalyData.anomaly_type}
                        </Typography>
                        <Typography variant="body2" color="white">
                          <strong>Severity:</strong> {anomalyData.severity}
                        </Typography>
                        <Typography variant="body2" color="white">
                          <strong>Score:</strong> {(anomalyData.score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Popup>
              </Marker>
              
              {isAnomaly && (
                <Circle
                  center={[flight.latitude, flight.longitude]}
                  radius={50000}
                  pathOptions={{
                    color: '#ff0000',
                    fillColor: '#ff0000',
                    fillOpacity: 0.1,
                    weight: 2
                  }}
                />
              )}
            </React.Fragment>
          )
        })}
      </MapContainer>
    </Box>
  )
}

export default MapView