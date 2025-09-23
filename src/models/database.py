# src/models/database.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
import asyncpg
import asyncio
from src.config.settings import settings

Base = declarative_base()

class BloggerPost(Base):
    __tablename__ = "blogger_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False)
    post_id = Column(String(100), unique=True, nullable=False)
    author = Column(String(100))
    content = Column(Text)
    quality_score = Column(Float, default=0.0)
    ai_parsed = Column(Boolean, default=False)
    extracted_signals = Column(JSON)
    ai_analysis = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

class AISignal(Base):
    __tablename__ = "ai_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String(100), nullable=False)
    symbol = Column(String(10), nullable=False)
    company_name = Column(String(100))
    action = Column(String(10))
    confidence = Column(Float)
    credibility_score = Column(Float)
    target_price = Column(Float)
    current_price = Column(Float)
    reasoning = Column(Text)
    risk_level = Column(String(20))
    time_horizon = Column(String(20))
    status = Column(String(20), default="PENDING")
    execution_method = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer)
    symbol = Column(String(10))
    action = Column(String(10))
    quantity = Column(Float)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    tradingview_alert_id = Column(String(100))
    status = Column(String(20))
    timestamp = Column(DateTime, default=datetime.utcnow)

class SystemMetrics(Base):
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    openai_requests = Column(Integer, default=0)
    openai_cost = Column(Float, default=0.0)
    signals_generated = Column(Integer, default=0)
    high_confidence_signals = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    email_alerts_sent = Column(Integer, default=0)

# SQLAlchemy sync engine removed - using async only

# 异步数据库访问（更高效）
class AsyncDatabaseManager:
    def __init__(self):
        self.pool = None
    
    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                settings.DATABASE_URL,
                min_size=1,      # 最小连接数
                max_size=5,      # 最大连接数，免费版限制
                command_timeout=60
            )
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()
    
    async def execute_query(self, query: str, *args):
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def fetch_all(self, query: str, *args):
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)

    async def execute_many(self, query: str, args_list):
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as connection:
            return await connection.executemany(query, args_list)

db_manager = AsyncDatabaseManager()