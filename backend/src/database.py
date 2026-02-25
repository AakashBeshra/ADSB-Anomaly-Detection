# Add this at the end of your existing database.py file
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
import pytz

from config import settings

Base = declarative_base()

class FlightData(Base):
    __tablename__ = "flight_data"
    
    id = Column(Integer, primary_key=True, index=True)
    icao24 = Column(String, index=True)
    callsign = Column(String, index=True)
    origin_country = Column(String)
    time_position = Column(Integer)
    last_contact = Column(Integer)
    longitude = Column(Float)
    latitude = Column(Float)
    altitude = Column(Float)
    velocity = Column(Float)
    heading = Column(Float)
    vertical_rate = Column(Float)
    sensors = Column(String)
    squawk = Column(String)
    spi = Column(Boolean)
    position_source = Column(Integer)
    category = Column(Integer)
    anomaly_score = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.now(pytz.UTC))

class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, index=True)
    anomaly_type = Column(String)
    severity = Column(String)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.now(pytz.UTC))

engine = create_async_engine(settings.DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session