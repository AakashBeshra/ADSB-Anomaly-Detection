import React, { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import Dashboard from './components/Dashboard'
import { io } from 'socket.io-client'
import { Box } from '@mui/material'

function App() {
  const [socket, setSocket] = useState(null)
  const [flights, setFlights] = useState([])
  const [anomalies, setAnomalies] = useState([])
  const [stats, setStats] = useState({})

  useEffect(() => {
    // Initialize WebSocket connection
    const newSocket = io('ws://localhost:8000', {
      path: '/ws',
      transports: ['websocket']
    })

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server')
    })

    newSocket.on('flight_update', (data) => {
      setFlights(data.flights || [])
      setAnomalies(data.anomalies || [])
      setStats(data.stats || {})
    })

    setSocket(newSocket)

    return () => {
      newSocket.close()
    }
  }, [])

  return (
    <Box sx={{ 
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0f1e 0%, #1a1f2e 100%)',
    }}>
      <Routes>
        <Route 
          path="/" 
          element={
            <Dashboard 
              flights={flights}
              anomalies={anomalies}
              stats={stats}
              socket={socket}
            />
          } 
        />
      </Routes>
    </Box>
  )
}

export default App