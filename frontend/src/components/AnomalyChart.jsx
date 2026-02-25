import React from 'react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { Box, Typography, ToggleButton, ToggleButtonGroup } from '@mui/material'
import { Timeline as TimelineIcon, BarChart as BarChartIcon } from '@mui/icons-material'
import { useState } from 'react'

const AnomalyChart = ({ anomalies }) => {
  const [chartType, setChartType] = useState('line')

  // Process anomaly data for charts
  const processAnomalyData = () => {
    const hourlyData = {}
    const typeData = {}
    const severityData = {
      high: 0,
      medium: 0,
      low: 0
    }

    anomalies.forEach(anomaly => {
      // Hourly distribution
      const hour = new Date(anomaly.timestamp).getHours()
      hourlyData[hour] = (hourlyData[hour] || 0) + 1

      // Type distribution
      const type = anomaly.anomaly_type
      typeData[type] = (typeData[type] || 0) + 1

      // Severity distribution
      severityData[anomaly.severity] = (severityData[anomaly.severity] || 0) + 1
    })

    // Convert to array format for charts
    const hourlyArray = Object.entries(hourlyData)
      .sort(([a], [b]) => a - b)
      .map(([hour, count]) => ({
        hour: `${hour}:00`,
        anomalies: count
      }))

    const typeArray = Object.entries(typeData)
      .map(([type, count]) => ({
        type,
        count
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5)

    const severityArray = Object.entries(severityData)
      .map(([severity, count]) => ({
        severity: severity.toUpperCase(),
        count,
        color: severity === 'high' ? '#ff4444' : 
               severity === 'medium' ? '#ffaa44' : '#44aa44'
      }))

    return { hourlyArray, typeArray, severityArray }
  }

  const { hourlyArray, typeArray, severityArray } = processAnomalyData()

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8']

  return (
    <Box>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 3 
      }}>
        <Typography variant="h6">
          Anomaly Analytics
        </Typography>
        
        <ToggleButtonGroup
          value={chartType}
          exclusive
          onChange={(_, value) => value && setChartType(value)}
          size="small"
        >
          <ToggleButton value="line">
            <TimelineIcon sx={{ mr: 1 }} />
            Timeline
          </ToggleButton>
          <ToggleButton value="bar">
            <BarChartIcon sx={{ mr: 1 }} />
            Distribution
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      <Box sx={{ height: 300 }}>
        {chartType === 'line' ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={hourlyArray}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="hour" 
                stroke="#888"
                tick={{ fill: '#888' }}
              />
              <YAxis 
                stroke="#888"
                tick={{ fill: '#888' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1a1f2e',
                  border: '1px solid #2d3748',
                  borderRadius: 4
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="anomalies"
                stroke="#ff4444"
                strokeWidth={2}
                dot={{ stroke: '#ff4444', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6 }}
                name="Anomaly Count"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={typeArray}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis 
                dataKey="type" 
                stroke="#888"
                tick={{ fill: '#888' }}
                angle={-45}
                textAnchor="end"
                height={60}
              />
              <YAxis 
                stroke="#888"
                tick={{ fill: '#888' }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1a1f2e',
                  border: '1px solid #2d3748',
                  borderRadius: 4
                }}
              />
              <Legend />
              <Bar 
                dataKey="count" 
                fill="#00b4d8"
                radius={[4, 4, 0, 0]}
                name="Anomaly Count"
              />
            </BarChart>
          </ResponsiveContainer>
        )}
      </Box>

      <Box sx={{ mt: 3, height: 200 }}>
        <Typography variant="subtitle2" sx={{ mb: 2, textAlign: 'center' }}>
          Severity Distribution
        </Typography>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={severityArray}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ severity, percent }) => `${severity}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="count"
            >
              {severityArray.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1a1f2e',
                border: '1px solid #2d3748',
                borderRadius: 4
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  )
}

export default AnomalyChart