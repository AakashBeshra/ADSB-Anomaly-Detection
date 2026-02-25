import React from 'react'
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Alert,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip
} from '@mui/material'
import {
  Radar as RadarIcon,
  Warning as WarningIcon,
  Flight as FlightIcon,
  Satellite as SatelliteIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon
} from '@mui/icons-material'
import MapView from './MapView'
import FlightTable from './FlightTable'
import AnomalyChart from './AnomalyChart'
import StatisticsPanel from './StatisticsPanel'
import AlertPanel from './AlertPanel'
import { motion } from 'framer-motion'

const Dashboard = ({ flights, anomalies, stats, socket }) => {
  const totalFlights = stats.total_flights || 0
  const totalAnomalies = stats.total_anomalies || 0
  const anomalyRate = stats.anomaly_rate || 0

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ 
        mb: 4, 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center' 
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <motion.div
            initial={{ rotate: 0 }}
            animate={{ rotate: 360 }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
          >
            <SatelliteIcon sx={{ fontSize: 40, color: '#00b4d8' }} />
          </motion.div>
          <Box>
            <Typography variant="h4" fontWeight="bold">
              ADS-B Anomaly Detection System
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Real-time flight monitoring with AI-powered anomaly detection
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={() => socket?.emit('refresh')}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)',
              color: 'white'
            }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <FlightIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Active Flights</Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold">
                  {totalFlights}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                  Real-time tracking
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #c62828 0%, #d32f2f 100%)',
              color: 'white'
            }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <WarningIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Anomalies</Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold">
                  {totalAnomalies}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                  {(anomalyRate * 100).toFixed(1)}% anomaly rate
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #2e7d32 0%, #388e3c 100%)',
              color: 'white'
            }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <RadarIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">Coverage</Typography>
                </Box>
                <Typography variant="h3" fontWeight="bold">
                  95%
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                  Global airspace coverage
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #6a1b9a 0%, #8e24aa 100%)',
              color: 'white'
            }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SatelliteIcon sx={{ mr: 1 }} />
                  <Typography variant="h6">System Status</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Chip 
                    label="OPERATIONAL" 
                    color="success" 
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  <LinearProgress 
                    sx={{ 
                      flexGrow: 1,
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: 'rgba(255,255,255,0.1)',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: '#4caf50'
                      }
                    }}
                    variant="determinate" 
                    value={100} 
                  />
                </Box>
                <Typography variant="body2" sx={{ mt: 1, opacity: 0.8 }}>
                  All systems nominal
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Main Content */}
      <Grid container spacing={3}>
        {/* Left Column - Map */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 2, height: '600px', borderRadius: 2 }}>
            <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
              <FlightIcon sx={{ mr: 1 }} />
              Real-time Flight Map
            </Typography>
            <MapView flights={flights} anomalies={anomalies} />
          </Paper>
        </Grid>

        {/* Right Column - Alerts & Stats */}
        <Grid item xs={12} lg={4}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, borderRadius: 2 }}>
                <AlertPanel anomalies={anomalies} />
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, borderRadius: 2 }}>
                <StatisticsPanel stats={stats} />
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper sx={{ p: 2, borderRadius: 2 }}>
                <AnomalyChart anomalies={anomalies} />
              </Paper>
            </Grid>
          </Grid>
        </Grid>

        {/* Flight Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, borderRadius: 2 }}>
            <FlightTable flights={flights} />
          </Paper>
        </Grid>
      </Grid>

      {/* Footer */}
      <Box sx={{ mt: 4, pt: 2, borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          ADS-B Anomaly Detection System v1.0 • Real-time monitoring powered by ML & DL • 
          Last update: {new Date().toLocaleTimeString()}
        </Typography>
      </Box>
    </Box>
  )
}

export default Dashboard