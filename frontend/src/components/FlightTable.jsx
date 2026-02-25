import React, { useState, useMemo } from 'react'
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  Chip,
  Typography,
  TextField,
  InputAdornment,
  Tooltip,
  IconButton
} from '@mui/material'
import {
  Search as SearchIcon,
  Warning as WarningIcon,
  Flight as FlightIcon,
  ArrowUpward as ArrowUpwardIcon,
  ArrowDownward as ArrowDownwardIcon
} from '@mui/icons-material'
import { visuallyHidden } from '@mui/utils'

const FlightTable = ({ flights }) => {
  const [page, setPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const [order, setOrder] = useState('desc')
  const [orderBy, setOrderBy] = useState('anomaly_score')
  const [search, setSearch] = useState('')

  const handleRequestSort = (property) => {
    const isAsc = orderBy === property && order === 'asc'
    setOrder(isAsc ? 'desc' : 'asc')
    setOrderBy(property)
  }

  const filteredFlights = useMemo(() => {
    return flights.filter(flight => {
      const searchLower = search.toLowerCase()
      return (
        (flight.callsign && flight.callsign.toLowerCase().includes(searchLower)) ||
        (flight.icao24 && flight.icao24.toLowerCase().includes(searchLower)) ||
        (flight.origin_country && flight.origin_country.toLowerCase().includes(searchLower))
      )
    })
  }, [flights, search])

  const sortedFlights = useMemo(() => {
    return [...filteredFlights].sort((a, b) => {
      let aVal = a[orderBy] || 0
      let bVal = b[orderBy] || 0
      
      if (orderBy === 'callsign' || orderBy === 'origin_country') {
        aVal = aVal || ''
        bVal = bVal || ''
        return order === 'asc' 
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal)
      }
      
      return order === 'asc' ? aVal - bVal : bVal - aVal
    })
  }, [filteredFlights, order, orderBy])

  return (
    <Box>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        mb: 3 
      }}>
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
          <FlightIcon sx={{ mr: 1 }} />
          Flight Data ({flights.length} total)
        </Typography>
        
        <TextField
          placeholder="Search flights..."
          size="small"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{ width: 300 }}
        />
      </Box>

      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'callsign'}
                  direction={orderBy === 'callsign' ? order : 'asc'}
                  onClick={() => handleRequestSort('callsign')}
                >
                  Callsign
                  {orderBy === 'callsign' ? (
                    <Box component="span" sx={visuallyHidden}>
                      {order === 'desc' ? 'sorted descending' : 'sorted ascending'}
                    </Box>
                  ) : null}
                </TableSortLabel>
              </TableCell>
              <TableCell>ICAO24</TableCell>
              <TableCell>Country</TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'altitude'}
                  direction={orderBy === 'altitude' ? order : 'asc'}
                  onClick={() => handleRequestSort('altitude')}
                >
                  Altitude (ft)
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'velocity'}
                  direction={orderBy === 'velocity' ? order : 'asc'}
                  onClick={() => handleRequestSort('velocity')}
                >
                  Speed (kts)
                </TableSortLabel>
              </TableCell>
              <TableCell>Heading</TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'anomaly_score'}
                  direction={orderBy === 'anomaly_score' ? order : 'asc'}
                  onClick={() => handleRequestSort('anomaly_score')}
                >
                  Anomaly Score
                </TableSortLabel>
              </TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          
          <TableBody>
            {sortedFlights
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((flight) => (
                <TableRow 
                  key={flight.icao24}
                  hover
                  sx={{ 
                    '&:last-child td, &:last-child th': { border: 0 },
                    bgcolor: flight.is_anomaly_ml ? 'rgba(255, 0, 0, 0.05)' : 'inherit'
                  }}
                >
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <FlightIcon sx={{ mr: 1, fontSize: 16, color: 'text.secondary' }} />
                      <Typography variant="body2" fontWeight="medium">
                        {flight.callsign || 'N/A'}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2" fontFamily="monospace">
                      {flight.icao24}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Chip 
                      label={flight.origin_country || 'Unknown'} 
                      size="small" 
                      variant="outlined"
                    />
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {flight.vertical_rate > 0 ? (
                        <ArrowUpwardIcon sx={{ fontSize: 16, color: 'success.main', mr: 0.5 }} />
                      ) : flight.vertical_rate < 0 ? (
                        <ArrowDownwardIcon sx={{ fontSize: 16, color: 'error.main', mr: 0.5 }} />
                      ) : null}
                      <Typography variant="body2">
                        {Math.round(flight.altitude || 0)}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    <Typography 
                      variant="body2" 
                      color={flight.velocity > 600 ? 'error.main' : 'inherit'}
                    >
                      {Math.round(flight.velocity || 0)}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Typography variant="body2">
                      {flight.heading ? `${Math.round(flight.heading)}°` : 'N/A'}
                    </Typography>
                  </TableCell>
                  
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Box sx={{ 
                        width: '100%',
                        bgcolor: 'rgba(255,255,255,0.1)',
                        borderRadius: 1,
                        height: 8,
                        mr: 1,
                        overflow: 'hidden'
                      }}>
                        <Box sx={{ 
                          width: `${(flight.anomaly_score || 0) * 100}%`,
                          height: '100%',
                          bgcolor: flight.anomaly_score > 0.8 ? '#ff4444' : 
                                  flight.anomaly_score > 0.6 ? '#ffaa44' : '#44ff44',
                          transition: 'width 0.3s ease'
                        }} />
                      </Box>
                      <Typography variant="body2" fontFamily="monospace">
                        {(flight.anomaly_score || 0).toFixed(3)}
                      </Typography>
                    </Box>
                  </TableCell>
                  
                  <TableCell>
                    {flight.is_anomaly_ml ? (
                      <Chip
                        icon={<WarningIcon />}
                        label="ANOMALY"
                        color="error"
                        size="small"
                        sx={{ fontWeight: 'bold' }}
                      />
                    ) : (
                      <Chip
                        label="NORMAL"
                        color="success"
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        rowsPerPageOptions={[5, 10, 25]}
        component="div"
        count={sortedFlights.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10))
          setPage(0)
        }}
      />
    </Box>
  )
}

export default FlightTable