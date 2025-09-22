# src/config/settings.py
import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "Free AI Trading System"
    DEBUG: bool = False
    
    # 数据库配置 - Supabase
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # OpenAI配置 - 成本控制
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4o-mini"  # 最便宜的模型
    OPENAI_MAX_TOKENS: int = 2000       # 减少token使用
    OPENAI_MONTHLY_BUDGET: float = 50.0 # 月度预算控制
    
    # TradingView配置
    TRADINGVIEW_WEBHOOK_URL: str = os.getenv("TRADINGVIEW_WEBHOOK_URL", "")
    
    # Gmail SMTP配置
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    ALERT_EMAIL: str = os.getenv("ALERT_EMAIL", "")
    
    # 交易配置
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_CREDIBILITY_SCORE: float = 0.6
    MAX_POSITION_SIZE: float = 1000.0
    STOP_LOSS_PERCENT: float = 0.05
    TAKE_PROFIT_PERCENT: float = 0.10
    
    # 监控配置
    BLOGGER_IDS: List[str] = ["blogger1", "blogger2"]
    CHECK_INTERVAL_MINUTES: int = 10  # 减少到10分钟，节省资源
    QUALITY_THRESHOLD: float = 0.6
    
    # 免费版限制
    MAX_DAILY_TRADES: int = 5        # 减少交易频率
    MAX_DAILY_LOSS: float = 200.0    # 降低风险限制
    MAX_MEMORY_MB: int = 450         # Railway免费版内存限制
    
    class Config:
        env_file = ".env"

settings = Settings()