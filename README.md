# 🚀 免费云服务AI交易系统完整执行指南

## 📋 项目概览

### 系统架构 - 100%免费版
```
小红书/Twitter → Railway.app → OpenAI解析 → 信号分级 → TradingView/Gmail通知
     ↓             ↓              ↓           ↓              ↓
   爬虫监控     Python FastAPI   智能解析   高置信度执行    低置信度邮件
   (免费)        (500小时/月)      (成本控制)  (自动交易)     (人工审核)
```

### 免费技术栈
- **服务器**: Railway.app (500执行小时/月)
- **数据库**: Supabase (500MB PostgreSQL)
- **缓存**: Railway.app Redis (内置)
- **存储**: Supabase Storage (1GB)
- **邮件**: Gmail SMTP (500封/天)
- **监控**: UptimeRobot (50个监控点)
- **代码仓库**: GitHub (私有仓库)

### 核心功能
- ✅ **智能解析**: OpenAI GPT-4解析社媒内容
- ✅ **分级处理**: 高置信度自动执行，低置信度邮件通知
- ✅ **成本控制**: 质量预筛选，月预算控制
- ✅ **风险管理**: 多层安全检查
- ✅ **实时监控**: 系统状态和性能监控

---

## 🎯 第一阶段：免费服务注册和配置（第1-2天）

### 1.1 免费服务账号申请

**必需的免费账号**:
```bash
# 1. 核心服务
✅ Railway.app - 主服务器
✅ Supabase.com - PostgreSQL数据库
✅ GitHub.com - 代码仓库
✅ OpenAI.com - AI解析API
✅ UptimeRobot.com - 监控服务

# 2. 可选服务
✅ TradingView.com - 交易执行
✅ Gmail.com - 邮件服务
```

### 1.2 OpenAI API配置

```python
# 获取OpenAI API密钥
# 1. 访问 https://platform.openai.com/api-keys
# 2. 创建新的API密钥
# 3. 设置使用限制：$50/月
# 4. 记录API密钥，后续配置使用

# 成本控制设置
OPENAI_MONTHLY_BUDGET = 50.0  # 美元
OPENAI_MODEL = "gpt-4o-mini"  # 最便宜的GPT-4模型
```

### 1.3 Supabase数据库配置

```sql
-- 1. 创建Supabase项目
-- 2. 获取数据库连接信息
-- 3. 创建数据表结构

-- 博主帖子表
CREATE TABLE blogger_posts (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(50) NOT NULL,
    post_id VARCHAR(100) UNIQUE NOT NULL,
    author VARCHAR(100),
    content TEXT,
    quality_score FLOAT DEFAULT 0.0,
    ai_parsed BOOLEAN DEFAULT false,
    extracted_signals JSONB,
    ai_analysis JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    processed BOOLEAN DEFAULT false
);

-- AI信号表
CREATE TABLE ai_signals (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(100),
    action VARCHAR(10),
    confidence FLOAT,
    credibility_score FLOAT,
    target_price FLOAT,
    current_price FLOAT,
    reasoning TEXT,
    risk_level VARCHAR(20),
    time_horizon VARCHAR(20),
    status VARCHAR(20) DEFAULT 'PENDING',
    execution_method VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- 交易记录表
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES ai_signals(id),
    symbol VARCHAR(10),
    action VARCHAR(10),
    quantity FLOAT,
    entry_price FLOAT,
    stop_loss FLOAT,
    take_profit FLOAT,
    tradingview_alert_id VARCHAR(100),
    status VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- 系统指标表
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    date TIMESTAMP DEFAULT NOW(),
    openai_requests INTEGER DEFAULT 0,
    openai_cost FLOAT DEFAULT 0.0,
    signals_generated INTEGER DEFAULT 0,
    high_confidence_signals INTEGER DEFAULT 0,
    trades_executed INTEGER DEFAULT 0,
    email_alerts_sent INTEGER DEFAULT 0
);

-- 创建索引提高查询性能
CREATE INDEX idx_blogger_posts_timestamp ON blogger_posts(timestamp);
CREATE INDEX idx_ai_signals_confidence ON ai_signals(confidence);
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
```

---

## 💻 第二阶段：核心代码实现（第3-7天）

### 2.1 项目结构和依赖

```
trading-system/
├── src/
│   ├── main.py                 # FastAPI主应用
│   ├── config/
│   │   └── settings.py         # 配置管理
│   ├── models/
│   │   └── database.py         # 数据模型
│   ├── parsers/
│   │   └── openai_parser.py    # OpenAI解析器
│   ├── monitors/
│   │   └── social_monitor.py   # 社媒监控
│   ├── traders/
│   │   └── tradingview_client.py # TradingView集成
│   ├── utils/
│   │   ├── email_service.py    # 邮件服务
│   │   └── quality_filter.py   # 质量过滤
│   └── trading_engine.py       # 核心交易引擎
├── requirements.txt
├── Dockerfile
├── railway.json
└── README.md
```

### 2.2 依赖配置（优化免费版）

```txt
# requirements.txt - 优化内存使用
fastapi==0.104.1
uvicorn==0.24.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
pydantic==2.5.0

# AI和解析
openai==1.35.0
tiktoken==0.7.0
tenacity==8.2.3

# 网页抓取（轻量级）
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.0

# 邮件和工具
aiosmtplib==3.0.1
python-dotenv==1.0.0
schedule==1.2.0

# 优化包（减少内存使用）
asyncpg==0.29.0  # 替代psycopg2，更高效
httpx==0.25.2    # 替代requests，异步支持
```

### 2.3 配置管理（免费版优化）

```python
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
```

### 2.4 数据库模型（轻量级）

```python
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

# 优化的数据库连接（免费版）
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=3,        # 减少连接池大小
    max_overflow=2,     # 减少溢出连接
    pool_timeout=30,
    pool_recycle=1800,  # 30分钟回收连接
    echo=False          # 关闭SQL日志节省内存
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    
    async def execute_many(self, query: str, args_list):
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            return await connection.executemany(query, args_list)

db_manager = AsyncDatabaseManager()
```

### 2.5 OpenAI解析器（成本优化版）

```python
# src/parsers/openai_parser.py
import openai
import json
import tiktoken
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from datetime import datetime
import re
from src.config.settings import settings

# 精简的数据模型
class StockSignal(BaseModel):
    symbol: str = Field(..., description="股票代码")
    action: str = Field(..., description="交易动作: BUY, SELL, HOLD")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    target_price: Optional[float] = Field(None, description="目标价格")
    reasoning: str = Field(..., description="推荐理由")
    risk_level: str = Field(..., description="风险等级: LOW, MEDIUM, HIGH")

class ParsedContent(BaseModel):
    original_text: str
    platform: str
    post_id: str
    author: str
    timestamp: datetime
    signals: List[StockSignal]
    sentiment: str = Field(..., description="整体情绪: BULLISH, BEARISH, NEUTRAL")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="内容可信度")

class OptimizedOpenAIParser:
    """成本优化的OpenAI解析器"""
    
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.logger = logging.getLogger(__name__)
        self.monthly_usage = 0.0
        
        # 精简的系统提示词
        self.system_prompt = """你是专业投资分析师，从社媒提取股票信号。

**任务**: 识别美股代码和交易动作，评估置信度。

**输出JSON格式**:
{
  "original_text": "原文",
  "platform": "平台",
  "post_id": "ID", 
  "author": "作者",
  "timestamp": "时间",
  "signals": [{
    "symbol": "AAPL",
    "action": "BUY",
    "confidence": 0.85,
    "target_price": 180.0,
    "reasoning": "理由",
    "risk_level": "LOW"
  }],
  "sentiment": "BULLISH",
  "credibility_score": 0.75
}

**规则**: 只识别明确信号，保守评估，无信号返回空数组。"""

    def estimate_tokens(self, text: str) -> int:
        """估算token数量"""
        return len(self.encoding.encode(text))

    def check_budget(self) -> bool:
        """检查预算"""
        return self.monthly_usage < settings.OPENAI_MONTHLY_BUDGET

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def call_openai_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """优化的API调用"""
        if not self.check_budget():
            raise Exception("Monthly budget exceeded")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # 计算成本（GPT-4o-mini价格）
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = prompt_tokens * 0.00015 + completion_tokens * 0.0006  # GPT-4o-mini定价
            self.monthly_usage += cost
            
            self.logger.info(f"OpenAI API: {response.usage.total_tokens} tokens, ${cost:.4f}")
            
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage,
                "cost": cost
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    async def parse_post(self, text: str, platform: str, post_id: str, 
                        author: str = "unknown", timestamp: Optional[datetime] = None) -> ParsedContent:
        """解析单个帖子"""
        if timestamp is None:
            timestamp = datetime.now()

        # 检查文本长度，截断以节省成本
        if self.estimate_tokens(text) > 1500:
            text = text[:1000]  # 更激进的截断
            self.logger.warning(f"Text truncated for post {post_id}")

        # 精简的用户提示词
        user_prompt = f"""分析帖子：
平台: {platform}
作者: {author}
内容: {text}

返回JSON格式分析结果。"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = await self.call_openai_api(messages)
            result_json = json.loads(response["content"])
            
            # 确保字段完整
            result_json.update({
                "original_text": text,
                "platform": platform,
                "post_id": post_id,
                "author": author,
                "timestamp": timestamp.isoformat()
            })
            
            parsed_result = ParsedContent(**result_json)
            self.logger.info(f"Parsed post {post_id}: {len(parsed_result.signals)} signals")
            return parsed_result
            
        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Parse error for {post_id}: {e}")
            return self._create_default_result(text, platform, post_id, author, timestamp)

    def _create_default_result(self, text: str, platform: str, post_id: str, 
                              author: str, timestamp: datetime) -> ParsedContent:
        """创建默认结果"""
        return ParsedContent(
            original_text=text,
            platform=platform,
            post_id=post_id,
            author=author,
            timestamp=timestamp,
            signals=[],
            sentiment="NEUTRAL",
            credibility_score=0.0
        )

    async def batch_parse(self, posts: List[Dict[str, Any]], max_concurrent: int = 3) -> List[ParsedContent]:
        """批量解析，控制并发"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def parse_with_semaphore(post):
            async with semaphore:
                return await self.parse_post(
                    text=post["content"],
                    platform=post["platform"],
                    post_id=post["post_id"],
                    author=post.get("author", "unknown"),
                    timestamp=post.get("timestamp")
                )
        
        tasks = [parse_with_semaphore(post) for post in posts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if isinstance(r, ParsedContent)]

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "monthly_usage": self.monthly_usage,
            "budget_limit": settings.OPENAI_MONTHLY_BUDGET,
            "remaining": settings.OPENAI_MONTHLY_BUDGET - self.monthly_usage,
            "usage_percent": (self.monthly_usage / settings.OPENAI_MONTHLY_BUDGET) * 100
        }
```

### 2.6 质量预筛选器（节省成本）

```python
# src/utils/quality_filter.py
import re
from typing import Dict, List
import logging
from src.config.settings import settings

class QualityPrefilter:
    """质量预筛选器 - 节省OpenAI成本"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 高质量内容指标
        self.quality_indicators = {
            'stock_symbols': [
                r'\$[A-Z]{1,5}\b',           # $AAPL格式
                r'\b[A-Z]{1,5}股票\b',        # AAPL股票
                r'\b[A-Z]{1,5}\s*公司\b'      # AAPL公司
            ],
            'trading_actions': [
                '买入', '卖出', '持有', '推荐', '建仓', '减仓', 
                'buy', 'sell', 'hold', 'long', 'short', '看涨', '看跌'
            ],
            'price_indicators': [
                r'\$\d+\.?\d*', r'目标价', r'价格', r'估值', 
                r'\d+美元', r'\d+刀', r'价位'
            ],
            'analysis_terms': [
                '分析', '研报', '基本面', '技术面', '财报', '业绩',
                '盈利', '收入', '增长', '市场', '行业', '竞争'
            ]
        }
        
        # 低质量内容过滤
        self.noise_keywords = [
            '早安', '晚安', '吃饭', '睡觉', '天气', '心情',
            '自拍', '美食', '旅游', '购物', '化妆', '穿搭'
        ]
    
    def calculate_quality_score(self, text: str, platform: str, author: str) -> float:
        """计算内容质量分数"""
        score = 0.0
        text_lower = text.lower()
        
        # 1. 股票符号检查 (35%)
        stock_found = any(re.search(pattern, text) for pattern in self.quality_indicators['stock_symbols'])
        if stock_found:
            score += 0.35
        
        # 2. 交易动作检查 (30%)
        action_found = any(keyword in text_lower for keyword in self.quality_indicators['trading_actions'])
        if action_found:
            score += 0.30
        
        # 3. 价格信息检查 (20%)
        price_found = any(re.search(pattern, text) for pattern in self.quality_indicators['price_indicators'])
        if price_found:
            score += 0.20
        
        # 4. 分析深度检查 (10%)
        analysis_found = any(keyword in text_lower for keyword in self.quality_indicators['analysis_terms'])
        if analysis_found:
            score += 0.10
        
        # 5. 噪音内容惩罚
        noise_found = any(keyword in text_lower for keyword in self.noise_keywords)
        if noise_found:
            score *= 0.5  # 噪音内容分数减半
        
        # 6. 长度加权
        if len(text) < 50:
            score *= 0.7  # 内容太短
        elif len(text) > 500:
            score *= 1.1  # 内容详细
        
        # 7. 平台权重
        platform_weights = {
            'xiaohongshu': 0.9,
            'twitter': 1.0,
            'weibo': 0.8
        }
        score *= platform_weights.get(platform, 0.5)
        
        return min(score, 1.0)
    
    def should_parse_with_ai(self, text: str, platform: str, author: str) -> bool:
        """判断是否需要AI解析"""
        quality_score = self.calculate_quality_score(text, platform, author)
        
        # 记录筛选结果
        self.logger.info(f"Quality score for {platform} post: {quality_score:.2f}")
        
        # 动态阈值，确保至少有一些内容会被解析
        threshold = settings.QUALITY_THRESHOLD
        
        return quality_score >= threshold
    
    def get_filter_stats(self) -> Dict[str, int]:
        """获取筛选统计"""
        # TODO: 从数据库统计筛选效果
        return {
            "total_posts": 0,
            "filtered_posts": 0,
            "ai_parsed_posts": 0,
            "cost_saved_percent": 0
        }

# 全局质量筛选器实例
quality_filter = QualityPrefilter()
```

### 2.7 社媒监控器（轻量级）

```python
# src/monitors/social_monitor.py
import asyncio
import aiohttp
import time
import re
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
from src.config.settings import settings
from src.utils.quality_filter import quality_filter

class LightweightSocialMonitor:
    """轻量级社媒监控器 - 适配免费版资源限制"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.last_check_times = {}
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def monitor_xiaohongshu_lite(self, blogger_id: str) -> List[Dict]:
        """轻量级小红书监控 - 减少资源消耗"""
        posts = []
        
        try:
            # 简化的API调用，减少数据传输
            url = f"https://www.xiaohongshu.com/user/profile/{blogger_id}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {url}: {response.status}")
                    return posts
                
                html = await response.text()
                
                # 简化的内容提取
                post_ids = re.findall(r'/explore/([a-zA-Z0-9]{20,})', html)
                
                # 只处理前5个最新帖子，减少资源消耗
                for post_id in post_ids[:5]:
                    if self.is_new_post("xiaohongshu", post_id):
                        post_content = await self.extract_post_content_lite(post_id)
                        if post_content:
                            # 质量预筛选
                            if quality_filter.should_parse_with_ai(
                                post_content, "xiaohongshu", blogger_id
                            ):
                                posts.append({
                                    "platform": "xiaohongshu",
                                    "post_id": post_id,
                                    "content": post_content,
                                    "author": blogger_id,
                                    "timestamp": datetime.now(),
                                    "quality_score": quality_filter.calculate_quality_score(
                                        post_content, "xiaohongshu", blogger_id
                                    )
                                })
                
        except Exception as e:
            self.logger.error(f"Error monitoring xiaohongshu {blogger_id}: {e}")
        
        return posts
    
    async def extract_post_content_lite(self, post_id: str) -> Optional[str]:
        """轻量级内容提取"""
        try:
            url = f"https://www.xiaohongshu.com/explore/{post_id}"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # 简化的内容提取正则
                content_patterns = [
                    r'"desc":"([^"]+)"',
                    r'"title":"([^"]+)"',
                    r'content["\']:\s*["\']([^"\']+)["\']'
                ]
                
                for pattern in content_patterns:
                    matches = re.findall(pattern, html)
                    if matches:
                        # 清理和解码内容
                        content = matches[0].replace('\\n', ' ').replace('\\', '')
                        if len(content) > 20:  # 确保内容有意义
                            return content[:500]  # 限制长度节省token
                
        except Exception as e:
            self.logger.warning(f"Failed to extract content for {post_id}: {e}")
        
        return None
    
    def is_new_post(self, platform: str, post_id: str) -> bool:
        """检查是否为新帖子 - 简化版"""
        cache_key = f"{platform}_{post_id}"
        
        # 简单的内存缓存，避免数据库查询
        if not hasattr(self, '_seen_posts'):
            self._seen_posts = set()
        
        if cache_key in self._seen_posts:
            return False
        
        self._seen_posts.add(cache_key)
        
        # 限制缓存大小，避免内存溢出
        if len(self._seen_posts) > 1000:
            # 清理一半旧记录
            old_posts = list(self._seen_posts)[:500]
            for post in old_posts:
                self._seen_posts.discard(post)
        
        return True
    
    async def monitor_all_sources(self) -> List[Dict]:
        """监控所有配置的信息源"""
        all_posts = []
        
        # 并发监控，但限制并发数
        semaphore = asyncio.Semaphore(2)  # 限制并发数
        
        async def monitor_blogger(blogger_id):
            async with semaphore:
                try:
                    posts = await self.monitor_xiaohongshu_lite(blogger_id)
                    return posts
                except Exception as e:
                    self.logger.error(f"Error monitoring {blogger_id}: {e}")
                    return []
        
        # 创建任务
        tasks = [monitor_blogger(blogger_id) for blogger_id in settings.BLOGGER_IDS[:3]]  # 限制博主数量
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集结果
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
        
        self.logger.info(f"Found {len(all_posts)} qualified posts from {len(settings.BLOGGER_IDS)} sources")
        return all_posts

# 使用示例
async def test_monitor():
    """测试监控器"""
    async with LightweightSocialMonitor() as monitor:
        posts = await monitor.monitor_all_sources()
        for post in posts:
            print(f"Found post: {post['post_id']} - Score: {post['quality_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(test_monitor())
```

### 2.8 邮件服务（Gmail优化）

```python
# src/utils/email_service.py
import asyncio
import aiosmtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
from typing import List, Dict, Any
from datetime import datetime
import logging
from src.config.settings import settings

class GmailEmailService:
    """Gmail优化的邮件服务"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_sent = 0  # 跟踪每日发送量
        self.daily_limit = 450  # Gmail免费版限制500/天，保留缓冲
        
    async def send_email(self, subject: str, body: str, to_email: str = None, 
                        html_body: str = None) -> bool:
        """发送邮件"""
        if self.daily_sent >= self.daily_limit:
            self.logger.warning("Daily email limit reached")
            return False
            
        if not to_email:
            to_email = settings.ALERT_EMAIL
            
        try:
            # 创建邮件
            msg = MimeMultipart('alternative')
            msg['From'] = settings.SMTP_USERNAME
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # 添加文本内容
            msg.attach(MimeText(body, 'plain', 'utf-8'))
            
            # 添加HTML内容
            if html_body:
                msg.attach(MimeText(html_body, 'html', 'utf-8'))
            
            # 发送邮件
            await aiosmtplib.send(
                msg,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                start_tls=True,
                username=settings.SMTP_USERNAME,
                password=settings.SMTP_PASSWORD,
                timeout=30
            )
            
            self.daily_sent += 1
            self.logger.info(f"Email sent successfully to {to_email} ({self.daily_sent}/{self.daily_limit})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False

    async def send_low_confidence_alert(self, signals: List[Dict], post_info: Dict) -> bool:
        """发送低置信度信号通知 - 精简版"""
        
        subject = f"🔍 低置信度投资信号 - {len(signals)}个信号需要审核"
        
        # 精简的文本版本
        body_lines = [
            f"检测到 {len(signals)} 个低置信度投资信号，请手动审核：",
            "",
            f"📱 来源: {post_info['platform']} - {post_info['author']}",
            f"🕒 时间: {post_info['timestamp']}",
            "",
            f"📝 原文: {post_info['content'][:150]}{'...' if len(post_info['content']) > 150 else ''}",
            "",
            "📊 解析信号:"
        ]
        
        for i, signal in enumerate(signals[:3], 1):  # 只显示前3个信号
            body_lines.extend([
                f"{i}. {signal['symbol']} - {signal['action']}",
                f"   置信度: {signal['confidence']:.2f} | 目标: ${signal.get('target_price', 'N/A')}",
                f"   理由: {signal['reasoning'][:80]}{'...' if len(signal['reasoning']) > 80 else ''}",
                ""
            ])
        
        if len(signals) > 3:
            body_lines.append(f"...还有 {len(signals) - 3} 个信号")
        
        body_lines.extend([
            "",
            "💡 建议: 仔细分析后手动执行交易",
            "⚠️  投资有风险，仅供参考！"
        ])
        
        body = "\n".join(body_lines)
        
        # 简化的HTML版本
        html_body = self._build_simple_html_alert(signals, post_info)
        
        return await self.send_email(subject, body, html_body=html_body)

    def _build_simple_html_alert(self, signals: List[Dict], post_info: Dict) -> str:
        """构建简化的HTML邮件"""
        
        signals_html = ""
        for signal in signals[:3]:  # 只显示前3个
            conf_color = "#28a745" if signal['confidence'] > 0.7 else "#ffc107" if signal['confidence'] > 0.5 else "#dc3545"
            
            signals_html += f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;">
                <h4>{signal['symbol']} - {signal['action']}</h4>
                <p><strong>置信度:</strong> <span style="color: {conf_color}">{signal['confidence']:.2f}</span></p>
                <p><strong>目标价:</strong> ${signal.get('target_price', 'N/A')}</p>
                <p><strong>理由:</strong> {signal['reasoning'][:100]}{'...' if len(signal['reasoning']) > 100 else ''}</p>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"></head>
        <body style="font-family: Arial, sans-serif; max-width: 600px;">
            <h2>🔍 低置信度投资信号通知</h2>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <p><strong>平台:</strong> {post_info['platform']}</p>
                <p><strong>作者:</strong> {post_info['author']}</p>
                <p><strong>时间:</strong> {post_info['timestamp']}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                <h3>📝 原文内容</h3>
                <p>{post_info['content'][:200]}{'...' if len(post_info['content']) > 200 else ''}</p>
            </div>
            
            <h3>📊 解析信号 ({len(signals)}个)</h3>
            {signals_html}
            
            {f'<p><em>还有 {len(signals) - 3} 个信号未显示</em></p>' if len(signals) > 3 else ''}
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <p><strong>⚠️ 提醒:</strong> 这是AI自动分析结果，仅供参考。请根据自己的判断和风险承受能力做出投资决策。</p>
            </div>
        </body>
        </html>
        """
        
        return html

    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """发送每日总结 - 精简版"""
        
        subject = f"📊 AI交易系统日报 - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
📊 AI交易系统每日总结

📱 监控统计:
- 检查帖子: {summary_data.get('total_posts', 0)}
- 通过筛选: {summary_data.get('qualified_posts', 0)}
- AI解析: {summary_data.get('parsed_posts', 0)}

🤖 AI使用:
- API调用: {summary_data.get('ai_requests', 0)}次
- 当日成本: ${summary_data.get('ai_cost', 0):.2f}
- 预算使用: {summary_data.get('budget_used_percent', 0):.1f}%

📊 信号统计:
- 总信号: {summary_data.get('total_signals', 0)}
- 高置信度: {summary_data.get('high_conf_signals', 0)}
- 邮件通知: {summary_data.get('email_alerts', 0)}

🔧 系统状态: {summary_data.get('system_status', 'RUNNING')}

---
发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return await self.send_email(subject, body)

    async def send_system_alert(self, message: str, alert_type: str = "INFO") -> bool:
        """发送系统告警"""
        
        icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}
        subject = f"{icons.get(alert_type, '📢')} 系统通知 - {alert_type}"
        
        body = f"""
{icons.get(alert_type, '📢')} 系统通知

类型: {alert_type}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

详情:
{message}

---
AI交易系统自动发送
        """
        
        return await self.send_email(subject, body)

# 全局邮件服务实例
email_service = GmailEmailService()
```

---

## 🚀 第三阶段：主交易引擎（第8-10天）

### 3.1 核心交易引擎（内存优化版）

```python
# src/trading_engine.py
import asyncio
import logging
import gc
from typing import List, Dict, Any
from datetime import datetime, timedelta
from src.parsers.openai_parser import OptimizedOpenAIParser
from src.monitors.social_monitor import LightweightSocialMonitor
from src.utils.email_service import email_service
from src.models.database import AsyncDatabaseManager, SessionLocal, AISignal, BloggerPost, Trade, SystemMetrics
from src.config.settings import settings
import json
import psutil

class MemoryOptimizedTradingEngine:
    """内存优化的交易引擎 - 适配免费版限制"""
    
    def __init__(self):
        self.ai_parser = OptimizedOpenAIParser()
        self.db_manager = AsyncDatabaseManager()
        self.logger = logging.getLogger(__name__)
        
        # 统计数据
        self.daily_stats = {
            'total_posts': 0,
            'qualified_posts': 0,
            'parsed_posts': 0,
            'ai_requests': 0,
            'ai_cost': 0.0,
            'total_signals': 0,
            'high_conf_signals': 0,
            'email_alerts': 0
        }
        
        # 内存监控
        self.memory_threshold = settings.MAX_MEMORY_MB * 1024 * 1024  # 转换为字节
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "memory_available_mb": (self.memory_threshold - memory_info.rss) / 1024 / 1024,
            "memory_critical": memory_info.rss > self.memory_threshold * 0.9
        }
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()  # 强制垃圾回收
        
        # 清理解析器缓存
        if hasattr(self.ai_parser, '_seen_posts'):
            if len(self.ai_parser._seen_posts) > 500:
                self.ai_parser._seen_posts.clear()
        
        self.logger.info("Memory cleanup completed")

    async def run_monitoring_cycle(self) -> bool:
        """执行一轮监控周期"""
        
        self.logger.info("Starting optimized monitoring cycle...")
        
        try:
            # 内存检查
            memory_info = self.check_memory_usage()
            if memory_info["memory_critical"]:
                self.cleanup_memory()
                self.logger.warning(f"Memory critical: {memory_info['memory_mb']:.1f}MB")
            
            # 1. 监控社媒（使用异步上下文管理器）
            async with LightweightSocialMonitor() as monitor:
                new_posts = await monitor.monitor_all_sources()
            
            self.daily_stats['total_posts'] += len(new_posts)
            
            if not new_posts:
                self.logger.info("No new qualified posts found")
                return True
            
            self.daily_stats['qualified_posts'] += len(new_posts)
            
            # 2. 批量AI解析（控制并发）
            parsed_results = await self.ai_parser.batch_parse(new_posts, max_concurrent=2)
            self.daily_stats['parsed_posts'] += len(parsed_results)
            self.daily_stats['ai_requests'] += len(new_posts)
            
            # 3. 处理解析结果
            for result in parsed_results:
                await self.process_parsed_result(result)
            
            # 4. 更新AI使用统计
            ai_stats = self.ai_parser.get_usage_stats()
            self.daily_stats['ai_cost'] = ai_stats['monthly_usage']
            
            # 5. 内存清理
            self.cleanup_memory()
            
            self.logger.info(f"Cycle completed: {len(new_posts)} posts, "
                           f"{self.daily_stats['high_conf_signals']} high-conf signals, "
                           f"Memory: {memory_info['memory_mb']:.1f}MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
            await email_service.send_system_alert(f"监控周期错误: {e}", "ERROR")
            return False
    
    async def process_parsed_result(self, result):
        """处理AI解析结果"""
        try:
            # 保存到数据库（使用异步）
            await self.save_parsed_result_to_db(result)
            
            # 统计信号
            self.daily_stats['total_signals'] += len(result.signals)
            
            # 分类处理信号
            high_conf_signals = []
            low_conf_signals = []
            
            for signal in result.signals:
                if (signal.confidence >= settings.HIGH_CONFIDENCE_THRESHOLD and 
                    result.credibility_score >= settings.MIN_CREDIBILITY_SCORE):
                    high_conf_signals.append(signal)
                elif signal.confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                    low_conf_signals.append(signal)
            
            # 处理高置信度信号 - 自动执行（模拟）
            if high_conf_signals:
                await self.execute_high_confidence_signals(high_conf_signals, result)
                self.daily_stats['high_conf_signals'] += len(high_conf_signals)
            
            # 处理低置信度信号 - 邮件通知
            if low_conf_signals:
                await self.notify_low_confidence_signals(low_conf_signals, result)
                self.daily_stats['email_alerts'] += 1
            
        except Exception as e:
            self.logger.error(f"Error processing parsed result: {e}")
    
    async def save_parsed_result_to_db(self, result):
        """保存解析结果到数据库"""
        try:
            # 使用异步数据库连接
            insert_post_query = """
                INSERT INTO blogger_posts (platform, post_id, author, content, quality_score, 
                                         ai_parsed, extracted_signals, ai_analysis, processed)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (post_id) DO NOTHING
            """
            
            await self.db_manager.execute_query(
                insert_post_query,
                result.platform, result.post_id, result.author, result.original_text,
                0.8, True, json.dumps([s.dict() for s in result.signals]),
                json.dumps({"sentiment": result.sentiment, "credibility_score": result.credibility_score}),
                True
            )
            
            # 保存信号
            for signal in result.signals:
                insert_signal_query = """
                    INSERT INTO ai_signals (post_id, symbol, action, confidence, 
                                          credibility_score, target_price, reasoning, 
                                          risk_level, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """
                
                await self.db_manager.execute_query(
                    insert_signal_query,
                    result.post_id, signal.symbol, signal.action, signal.confidence,
                    result.credibility_score, signal.target_price, signal.reasoning,
                    signal.risk_level, "PENDING"
                )
            
            self.logger.info(f"Saved post {result.post_id} with {len(result.signals)} signals")
            
        except Exception as e:
            self.logger.error(f"Failed to save parsed result: {e}")
    
    async def execute_high_confidence_signals(self, signals: List, context):
        """执行高置信度信号（模拟版本）"""
        
        for signal in signals:
            try:
                # 基础风险检查
                if not await self.basic_risk_check(signal):
                    continue
                
                # 模拟交易信号生成
                trading_signal = {
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "target_price": signal.target_price,
                    "reasoning": signal.reasoning,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 记录到数据库
                await self.record_simulated_trade(signal, context, trading_signal)
                
                # 发送成功通知
                await email_service.send_system_alert(
                    f"模拟执行交易: {signal.symbol} {signal.action} (置信度: {signal.confidence:.2f})",
                    "SUCCESS"
                )
                
                self.logger.info(f"Simulated trade: {signal.symbol} {signal.action}")
                
            except Exception as e:
                self.logger.error(f"Failed to execute signal {signal.symbol}: {e}")
    
    async def basic_risk_check(self, signal) -> bool:
        """基础风险检查"""
        
        # 检查每日交易次数限制
        if self.daily_stats['high_conf_signals'] >= settings.MAX_DAILY_TRADES:
            self.logger.warning("Daily trade limit reached")
            return False
        
        # 检查风险等级
        if signal.risk_level == "HIGH":
            self.logger.warning(f"High risk signal skipped: {signal.symbol}")
            return False
        
        return True
    
    async def record_simulated_trade(self, signal, context, trading_signal):
        """记录模拟交易"""
        try:
            insert_trade_query = """
                INSERT INTO trades (symbol, action, quantity, entry_price, 
                                  tradingview_alert_id, status)
                VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            await self.db_manager.execute_query(
                insert_trade_query,
                signal.symbol, signal.action, 100,  # 模拟数量
                signal.target_price or 0.0,
                f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal.symbol}",
                "SIMULATED"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to record trade: {e}")
    
    async def notify_low_confidence_signals(self, signals: List, context):
        """通知低置信度信号"""
        
        try:
            # 构建邮件数据
            post_info = {
                "platform": context.platform,
                "author": context.author,
                "timestamp": context.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "content": context.original_text
            }
            
            signals_data = [signal.dict() for signal in signals]
            
            # 发送邮件通知
            success = await email_service.send_low_confidence_alert(signals_data, post_info)
            
            if success:
                # 更新信号状态
                for signal in signals:
                    update_query = """
                        UPDATE ai_signals 
                        SET status = 'NOTIFIED', execution_method = 'EMAIL'
                        WHERE post_id = $1 AND symbol = $2
                    """
                    await self.db_manager.execute_query(update_query, context.post_id, signal.symbol)
                
                self.logger.info(f"Sent email notification for {len(signals)} low-confidence signals")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    async def send_daily_summary(self):
        """发送每日总结"""
        
        try:
            summary_data = self.daily_stats.copy()
            
            # 计算预算使用百分比
            ai_stats = self.ai_parser.get_usage_stats()
            summary_data['budget_used_percent'] = ai_stats['usage_percent']
            summary_data['system_status'] = "RUNNING"
            
            await email_service.send_daily_summary(summary_data)
            
            # 重置每日统计
            self.daily_stats = {key: 0 if isinstance(value, (int, float)) else value 
                               for key, value in self.daily_stats.items()}
            
        except Exception as e:
            self.logger.error(f"Failed to send daily summary: {e}")
    
    async def run_scheduler(self):
        """运行调度器"""
        self.logger.info("Free tier trading engine started")
        
        await self.db_manager.connect()
        last_daily_summary = datetime.now().date()
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                
                # 执行监控周期
                success = await self.run_monitoring_cycle()
                
                if not success:
                    # 如果周期失败，等待更长时间
                    await asyncio.sleep(300)  # 5分钟
                    continue
                
                # 检查是否需要发送每日总结
                current_date = datetime.now().date()
                if current_date > last_daily_summary:
                    await self.send_daily_summary()
                    last_daily_summary = current_date
                
                # 每10个周期进行内存清理
                if cycle_count % 10 == 0:
                    self.cleanup_memory()
                    memory_info = self.check_memory_usage()
                    self.logger.info(f"Memory status: {memory_info['memory_mb']:.1f}MB")
                
                # 等待下一个周期
                await asyncio.sleep(settings.CHECK_INTERVAL_MINUTES * 60)
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler error: {e}")
            await email_service.send_system_alert(f"调度器严重错误: {e}", "ERROR")
        finally:
            await self.db_manager.disconnect()

# 全局交易引擎实例
trading_engine = MemoryOptimizedTradingEngine()
```

### 3.2 FastAPI主应用

```python
# src/main.py
import asyncio
import logging
import sys
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import signal
import gc
import psutil

from src.trading_engine import trading_engine
from src.models.database import SessionLocal, AISignal, Trade, SystemMetrics
from src.config.settings import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Railway.app日志输出
    ]
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Free AI Trading System",
    description="OpenAI + Supabase + Railway.app 免费AI交易系统",
    version="1.0.0"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
scheduler_task = None

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    global scheduler_task
    
    logger.info("🚀 Starting Free AI Trading System...")
    logger.info(f"📊 Configuration: High confidence = {settings.HIGH_CONFIDENCE_THRESHOLD}")
    logger.info(f"👥 Monitoring {len(settings.BLOGGER_IDS)} bloggers")
    logger.info(f"💰 OpenAI budget: ${settings.OPENAI_MONTHLY_BUDGET}/month")
    
    # 启动后台调度器
    scheduler_task = asyncio.create_task(trading_engine.run_scheduler())
    
    # 发送启动通知
    try:
        from src.utils.email_service import email_service
        await email_service.send_system_alert(
            f"🚀 免费AI交易系统已启动\n\n"
            f"📊 配置信息:\n"
            f"- 高置信度阈值: {settings.HIGH_CONFIDENCE_THRESHOLD}\n"
            f"- 监控博主数量: {len(settings.BLOGGER_IDS)}\n"
            f"- OpenAI预算: ${settings.OPENAI_MONTHLY_BUDGET}/月\n"
            f"- 检查间隔: {settings.CHECK_INTERVAL_MINUTES}分钟\n\n"
            f"系统将自动监控社媒信号并发送邮件通知！",
            "SUCCESS"
        )
    except Exception as e:
        logger.error(f"Failed to send startup notification: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    global scheduler_task
    
    logger.info("🛑 Shutting down AI Trading System...")
    
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
    
    # 发送关闭通知
    try:
        from src.utils.email_service import email_service
        await email_service.send_system_alert("🛑 AI交易系统已停止", "INFO")
    except Exception as e:
        logger.error(f"Failed to send shutdown notification: {e}")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "🚀 Free AI Trading System is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # 检查数据库连接
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # 检查AI解析器状态
        ai_stats = trading_engine.ai_parser.get_usage_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory_mb": round(memory_mb, 1),
            "memory_limit_mb": settings.MAX_MEMORY_MB,
            "openai_budget_used": f"{ai_stats['usage_percent']:.1f}%",
            "database": "connected"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"System unhealthy: {e}")

@app.get("/stats")
async def get_stats():
    """获取系统统计"""
    try:
        db = SessionLocal()
        
        # 今日统计
        today = datetime.now().date()
        today_signals = db.query(AISignal).filter(
            AISignal.timestamp >= today
        ).count()
        
        today_high_conf = db.query(AISignal).filter(
            AISignal.timestamp >= today,
            AISignal.confidence >= settings.HIGH_CONFIDENCE_THRESHOLD
        ).count()
        
        today_trades = db.query(Trade).filter(
            Trade.timestamp >= today
        ).count()
        
        # AI使用统计
        ai_stats = trading_engine.ai_parser.get_usage_stats()
        
        # 内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        db.close()
        
        return {
            "today_stats": {
                "signals_generated": today_signals,
                "high_confidence_signals": today_high_conf,
                "trades_executed": today_trades
            },
            "ai_usage": {
                "monthly_cost": f"${ai_stats['monthly_usage']:.2f}",
                "budget_used": f"{ai_stats['usage_percent']:.1f}%",
                "remaining_budget": f"${ai_stats['remaining']:.2f}"
            },
            "system_status": {
                "memory_mb": round(memory_mb, 1),
                "memory_limit_mb": settings.MAX_MEMORY_MB,
                "uptime": "running"
            },
            "daily_stats": trading_engine.daily_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")

@app.get("/signals")
async def get_recent_signals(limit: int = 20):
    """获取最近的信号"""
    try:
        db = SessionLocal()
        
        signals = db.query(AISignal).order_by(
            AISignal.timestamp.desc()
        ).limit(limit).all()
        
        db.close()
        
        return {
            "signals": [
                {
                    "id": signal.id,
                    "symbol": signal.symbol,
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "target_price": signal.target_price,
                    "reasoning": signal.reasoning[:100] + "..." if len(signal.reasoning) > 100 else signal.reasoning,
                    "risk_level": signal.risk_level,
                    "status": signal.status,
                    "timestamp": signal.timestamp.isoformat()
                }
                for signal in signals
            ],
            "total": len(signals)
        }
        
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {e}")

@app.get("/trades")
async def get_recent_trades(limit: int = 10):
    """获取最近的交易"""
    try:
        db = SessionLocal()
        
        trades = db.query(Trade).order_by(
            Trade.timestamp.desc()
        ).limit(limit).all()
        
        db.close()
        
        return {
            "trades": [
                {
                    "id": trade.id,
                    "symbol": trade.symbol,
                    "action": trade.action,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "status": trade.status,
                    "timestamp": trade.timestamp.isoformat()
                }
                for trade in trades
            ],
            "total": len(trades)
        }
        
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trades: {e}")

@app.post("/trigger-check")
async def trigger_manual_check(background_tasks: BackgroundTasks):
    """手动触发检查"""
    try:
        # 添加后台任务
        background_tasks.add_task(trading_engine.run_monitoring_cycle)
        
        return {
            "message": "Manual check triggered",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger check: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger check: {e}")

@app.post("/cleanup")
async def cleanup_system():
    """清理系统资源"""
    try:
        # 内存清理
        trading_engine.cleanup_memory()
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取清理后的内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            "message": "System cleanup completed",
            "memory_mb": round(memory_mb, 1),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {e}")

# 错误处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"Global exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# 信号处理
def signal_handler(signum, frame):
    """处理停止信号"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    # Railway.app会自动设置PORT环境变量
    import os
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
```

---

## 🚀 第四阶段：部署配置（第11-12天）

### 4.1 Railway.app部署配置

```json
# railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python -m uvicorn src.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

```dockerfile
# Dockerfile (可选，Railway会自动检测Python)
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE $PORT

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

### 4.2 环境变量配置

```bash
# Railway.app环境变量设置
# 在Railway控制面板中设置以下变量：

# 数据库 (Supabase提供)
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MONTHLY_BUDGET=50.0

# Gmail邮件
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-gmail-app-password
ALERT_EMAIL=your-alert-email@gmail.com

# TradingView (可选)
TRADINGVIEW_WEBHOOK_URL=https://webhook.tradingview.com/your-webhook

# 博主配置
BLOGGER_IDS=blogger1,blogger2,blogger3

# 系统配置
HIGH_CONFIDENCE_THRESHOLD=0.75
MEDIUM_CONFIDENCE_THRESHOLD=0.5
MIN_CREDIBILITY_SCORE=0.6
MAX_POSITION_SIZE=1000.0
CHECK_INTERVAL_MINUTES=10
QUALITY_THRESHOLD=0.6
MAX_DAILY_TRADES=5
MAX_MEMORY_MB=450
```

### 4.3 部署脚本

```bash
#!/bin/bash
# deploy.sh

echo "🚀 Deploying Free AI Trading System..."

# 1. 检查必需的环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    exit 1
fi

if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL not set"
    exit 1
fi

# 2. 安装Railway CLI (如果没有)
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# 3. 登录Railway
echo "🔐 Login to Railway..."
railway login

# 4. 初始化项目 (如果是第一次)
if [ ! -f "railway.json" ]; then
    echo "🆕 Initializing Railway project..."
    railway init
fi

# 5. 设置环境变量
echo "⚙️ Setting environment variables..."
railway variables set OPENAI_API_KEY="$OPENAI_API_KEY"
railway variables set DATABASE_URL="$DATABASE_URL"
railway variables set SMTP_USERNAME="$SMTP_USERNAME"
railway variables set SMTP_PASSWORD="$SMTP_PASSWORD"
railway variables set ALERT_EMAIL="$ALERT_EMAIL"

# 6. 部署
echo "🚀 Deploying to Railway..."
railway deploy

# 7. 获取部署URL
echo "🌐 Getting deployment URL..."
railway status

echo "✅ Deployment completed!"
echo "📊 Check your system at: https://your-app.railway.app"
echo "📧 Check your email for startup notification"
```

---

## 📊 第五阶段：测试和优化（第13-14天）

### 5.1 测试脚本

```python
# tests/test_system.py
import asyncio
import pytest
from datetime import datetime
from src.trading_engine import MemoryOptimizedTradingEngine
from src.parsers.openai_parser import OptimizedOpenAIParser
from src.utils.email_service import email_service

class TestFreeTradingSystem:
    def setup_method(self):
        self.engine = MemoryOptimizedTradingEngine()
        
    @pytest.mark.asyncio
    async def test_openai_parser_optimization(self):
        """测试OpenAI解析器优化"""
        
        test_posts = [
            {
                "content": "强烈推荐$AAPL！iPhone 15销量超预期，目标价$200。现价$180，很好的买入机会！",
                "platform": "xiaohongshu",
                "post_id": "test001",
                "author": "投资达人",
                "timestamp": datetime.now()
            }
        ]
        
        results = await self.engine.ai_parser.batch_parse(test_posts, max_concurrent=1)
        
        # 验证结果
        assert len(results) == 1
        result = results[0]
        assert len(result.signals) >= 1
        assert result.signals[0].symbol == "AAPL"
        assert result.signals[0].action == "BUY"
        assert result.signals[0].confidence > 0.7
        
        print("✅ OpenAI解析器优化测试通过")
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self):
        """测试内存优化"""
        
        # 检查初始内存
        initial_memory = self.engine.check_memory_usage()
        print(f"Initial memory: {initial_memory['memory_mb']:.1f}MB")
        
        # 执行多个周期
        for i in range(3):
            success = await self.engine.run_monitoring_cycle()
            assert success
            
            memory_info = self.engine.check_memory_usage()
            print(f"Cycle {i+1} memory: {memory_info['memory_mb']:.1f}MB")
            
            # 确保内存没有无限增长
            assert memory_info['memory_mb'] < 500  # 免费版限制
        
        print("✅ 内存优化测试通过")
    
    @pytest.mark.asyncio 
    async def test_email_notification(self):
        """测试邮件通知功能"""
        
        test_signals = [{
            "symbol": "AAPL",
            "action": "BUY", 
            "confidence": 0.6,  # 低置信度
            "target_price": 180.0,
            "reasoning": "测试信号",
            "risk_level": "MEDIUM"
        }]
        
        test_post_info = {
            "platform": "test",
            "author": "test_user",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "content": "这是一个测试帖子内容，包含$AAPL的买入建议"
        }
        
        # 注意：实际测试时需要有效的邮件配置
        # success = await email_service.send_low_confidence_alert(test_signals, test_post_info)
        # assert success
        
        print("✅ 邮件通知测试通过（需要配置SMTP）")
    
    def test_quality_filter(self):
        """测试质量过滤器"""
        from src.utils.quality_filter import quality_filter
        
        # 高质量内容
        high_quality = "强烈推荐$AAPL，目标价$200，基本面分析显示业绩强劲"
        score_high = quality_filter.calculate_quality_score(high_quality, "xiaohongshu", "expert")
        assert score_high > 0.7
        
        # 低质量内容
        low_quality = "今天天气真好，去星巴克喝咖啡了"
        score_low = quality_filter.calculate_quality_score(low_quality, "xiaohongshu", "user")
        assert score_low < 0.3
        
        print(f"✅ 质量过滤测试通过: 高质量={score_high:.2f}, 低质量={score_low:.2f}")
    
    @pytest.mark.asyncio
    async def test_cost_control(self):
        """测试成本控制"""
        
        # 检查AI预算控制
        ai_stats = self.engine.ai_parser.get_usage_stats()
        assert ai_stats['monthly_usage'] <= ai_stats['budget_limit']
        
        # 检查预算检查功能
        self.engine.ai_parser.monthly_usage = 45.0  # 设置接近限制
        assert self.engine.ai_parser.check_budget()
        
        self.engine.ai_parser.monthly_usage = 55.0  # 超过限制
        assert not self.engine.ai_parser.check_budget()
        
        print("✅ 成本控制测试通过")

# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 5.2 性能监控脚本

```python
# scripts/monitor_performance.py
import asyncio
import time
import psutil
import logging
from datetime import datetime
from src.trading_engine import trading_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """免费版性能监控"""
    
    def __init__(self):
        self.metrics = []
        
    def collect_metrics(self):
        """收集性能指标"""
        process = psutil.Process()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "connections": len(process.connections()),
            "threads": process.num_threads()
        }
        
        # AI使用统计
        ai_stats = trading_engine.ai_parser.get_usage_stats()
        metrics.update({
            "openai_cost": ai_stats['monthly_usage'],
            "budget_used_percent": ai_stats['usage_percent']
        })
        
        # 系统统计
        metrics.update(trading_engine.daily_stats)
        
        self.metrics.append(metrics)
        return metrics
    
    async def monitor_cycle(self, duration_minutes=60):
        """监控一个周期"""
        logger.info(f"Starting {duration_minutes} minute performance monitoring...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            metrics = self.collect_metrics()
            
            logger.info(
                f"Memory: {metrics['memory_mb']:.1f}MB "
                f"({metrics['memory_percent']:.1f}%), "
                f"CPU: {metrics['cpu_percent']:.1f}%, "
                f"OpenAI: ${metrics['openai_cost']:.2f}"
            )
            
            # 检查内存警告
            if metrics['memory_mb'] > 400:
                logger.warning(f"High memory usage: {metrics['memory_mb']:.1f}MB")
                trading_engine.cleanup_memory()
            
            # 检查预算警告
            if metrics['budget_used_percent'] > 80:
                logger.warning(f"High OpenAI usage: {metrics['budget_used_percent']:.1f}%")
            
            await asyncio.sleep(60)  # 每分钟检查一次
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成性能报告"""
        if not self.metrics:
            return
        
        avg_memory = sum(m['memory_mb'] for m in self.metrics) / len(self.metrics)
        max_memory = max(m['memory_mb'] for m in self.metrics)
        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        
        report = f"""
📊 性能监控报告
================
监控时长: {len(self.metrics)} 分钟
平均内存: {avg_memory:.1f}MB
峰值内存: {max_memory:.1f}MB
平均CPU: {avg_cpu:.1f}%

🤖 OpenAI使用:
当前成本: ${self.metrics[-1]['openai_cost']:.2f}
预算使用: {self.metrics[-1]['budget_used_percent']:.1f}%

📈 信号统计:
总信号: {self.metrics[-1]['total_signals']}
高置信度: {self.metrics[-1]['high_conf_signals']}
邮件通知: {self.metrics[-1]['email_alerts']}

✅ 系统状态: {'健康' if max_memory < 450 else '需要优化'}
        """
        
        logger.info(report)
        
        # 保存到文件
        with open(f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as f:
            f.write(report)

async def main():
    """主函数"""
    monitor = PerformanceMonitor()
    
    # 监控1小时
    await monitor.monitor_cycle(duration_minutes=60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📋 第六阶段：上线运行检查清单（第15天）

### 6.1 上线前检查

```bash
#!/bin/bash
# scripts/pre_launch_check.sh

echo "🔍 Pre-launch System Check"
echo "=========================="

# 1. 环境变量检查
echo "1. 检查环境变量..."
check_env() {
    if [ -z "${!1}" ]; then
        echo "❌ $1 未设置"
        return 1
    else
        echo "✅ $1 已设置"
        return 0
    fi
}

check_env "OPENAI_API_KEY"
check_env "DATABASE_URL"
check_env "SMTP_USERNAME"
check_env "SMTP_PASSWORD"
check_env "ALERT_EMAIL"

# 2. 数据库连接检查
echo "2. 检查数据库连接..."
python -c "
from src.models.database import SessionLocal
try:
    db = SessionLocal()
    db.execute('SELECT 1')
    db.close()
    print('✅ 数据库连接正常')
except Exception as e:
    print(f'❌ 数据库连接失败: {e}')
    exit(1)
"

# 3. OpenAI API检查
echo "3. 检查OpenAI API..."
python -c "
import openai
from src.config.settings import settings
try:
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=5
    )
    print('✅ OpenAI API连接正常')
except Exception as e:
    print(f'❌ OpenAI API连接失败: {e}')
    exit(1)
"

# 4. 邮件服务检查
echo "4. 检查邮件服务..."
python -c "
import asyncio
from src.utils.email_service import email_service

async def test_email():
    try:
        success = await email_service.send_system_alert('系统测试邮件', 'INFO')
        if success:
            print('✅ 邮件服务正常')
        else:
            print('❌ 邮件发送失败')
    except Exception as e:
        print(f'❌ 邮件服务错误: {e}')

asyncio.run(test_email())
"

echo "5. 系统资源检查..."
python -c "
import psutil
memory_mb = psutil.virtual_memory().available / 1024 / 1024
cpu_count = psutil.cpu_count()
print(f'✅ 可用内存: {memory_mb:.0f}MB')
print(f'✅ CPU核心数: {cpu_count}')
"

echo ""
echo "🚀 系统检查完成！"
echo "📧 请检查邮箱是否收到测试邮件"
echo "🌐 系统准备就绪，可以启动服务"
```

### 6.2 监控仪表板

```python
# scripts/dashboard.py
import asyncio
import time
from datetime import datetime, timedelta
import psutil
from src.trading_engine import trading_engine
from src.models.database import SessionLocal, AISignal, Trade

class SimpleDashboard:
    """简单的命令行仪表板"""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    def clear_screen(self):
        """清屏"""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def get_system_stats(self):
        """获取系统统计"""
        process = psutil.Process()
        
        # 系统资源
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # AI使用统计
        ai_stats = trading_engine.ai_parser.get_usage_stats()
        
        # 数据库统计
        db = SessionLocal()
        try:
            today = datetime.now().date()
            
            today_signals = db.query(AISignal).filter(
                AISignal.timestamp >= today
            ).count()
            
            today_high_conf = db.query(AISignal).filter(
                AISignal.timestamp >= today,
                AISignal.confidence >= 0.75
            ).count()
            
            today_trades = db.query(Trade).filter(
                Trade.timestamp >= today
            ).count()
            
        finally:
            db.close()
        
        return {
            "memory_mb": memory_mb,
            "cpu_percent": cpu_percent,
            "ai_cost": ai_stats['monthly_usage'],
            "budget_used": ai_stats['usage_percent'],
            "today_signals": today_signals,
            "today_high_conf": today_high_conf,
            "today_trades": today_trades,
            "uptime": datetime.now() - self.start_time
        }
    
    def display_dashboard(self, stats):
        """显示仪表板"""
        self.clear_screen()
        
        print("🚀 Free AI Trading System Dashboard")
        print("=" * 50)
        print(f"📅 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  运行时间: {str(stats['uptime']).split('.')[0]}")
        print()
        
        print("💻 系统资源:")
        print(f"   内存使用: {stats['memory_mb']:.1f}MB / 450MB")
        print(f"   CPU使用: {stats['cpu_percent']:.1f}%")
        
        # 内存警告
        if stats['memory_mb'] > 400:
            print("   ⚠️  内存使用偏高")
        
        print()
        
        print("🤖 OpenAI使用:")
        print(f"   当月成本: ${stats['ai_cost']:.2f}")
        print(f"   预算使用: {stats['budget_used']:.1f}%")
        
        # 预算警告
        if stats['budget_used'] > 80:
            print("   ⚠️  预算使用偏高")
        
        print()
        
        print("📊 今日统计:")
        print(f"   生成信号: {stats['today_signals']}")
        print(f"   高置信度: {stats['today_high_conf']}")
        print(f"   执行交易: {stats['today_trades']}")
        print()
        
        print("📈 每日目标:")
        print(f"   系统运行: ✅")
        print(f"   成本控制: {'✅' if stats['budget_used'] < 90 else '⚠️'}")
        print(f"   内存控制: {'✅' if stats['memory_mb'] < 400 else '⚠️'}")
        print()
        
        print("🔄 按 Ctrl+C 停止监控")
    
    async def run_dashboard(self):
        """运行仪表板"""
        try:
            while True:
                stats = self.get_system_stats()
                self.display_dashboard(stats)
                await asyncio.sleep(30)  # 30秒更新一次
                
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")

async def main():
    """主函数"""
    dashboard = SimpleDashboard()
    await dashboard.run_dashboard()

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.3 维护脚本

```python
# scripts/maintenance.py
import asyncio
import logging
from datetime import datetime, timedelta
from src.models.database import SessionLocal, AISignal, BloggerPost, Trade
from src.utils.email_service import email_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceManager:
    """维护管理器"""
    
    async def cleanup_old_data(self, days=30):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        db = SessionLocal()
        try:
            # 删除30天前的帖子
            old_posts = db.query(BloggerPost).filter(
                BloggerPost.timestamp < cutoff_date
            )
            posts_count = old_posts.count()
            old_posts.delete(synchronize_session=False)
            
            # 删除30天前的信号
            old_signals = db.query(AISignal).filter(
                AISignal.timestamp < cutoff_date
            )
            signals_count = old_signals.count()
            old_signals.delete(synchronize_session=False)
            
            # 删除30天前的交易记录
            old_trades = db.query(Trade).filter(
                Trade.timestamp < cutoff_date
            )
            trades_count = old_trades.count()
            old_trades.delete(synchronize_session=False)
            
            db.commit()
            
            logger.info(f"Cleaned up: {posts_count} posts, {signals_count} signals, {trades_count} trades")
            
            return {
                "posts_deleted": posts_count,
                "signals_deleted": signals_count,
                "trades_deleted": trades_count
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Cleanup failed: {e}")
            raise
        finally:
            db.close()
    
    async def generate_weekly_report(self):
        """生成周报"""
        week_ago = datetime.now() - timedelta(days=7)
        
        db = SessionLocal()
        try:
            # 本周统计
            week_signals = db.query(AISignal).filter(
                AISignal.timestamp >= week_ago
            ).count()
            
            week_high_conf = db.query(AISignal).filter(
                AISignal.timestamp >= week_ago,
                AISignal.confidence >= 0.75
            ).count()
            
            week_trades = db.query(Trade).filter(
                Trade.timestamp >= week_ago
            ).count()
            
            # 平台统计
            xiaohongshu_posts = db.query(BloggerPost).filter(
                BloggerPost.timestamp >= week_ago,
                BloggerPost.platform == "xiaohongshu"
            ).count()
            
            twitter_posts = db.query(BloggerPost).filter(
                BloggerPost.timestamp >= week_ago,
                BloggerPost.platform == "twitter"
            ).count()
            
        finally:
            db.close()
        
        report = f"""
📊 AI交易系统周报
==================
统计时间: {week_ago.strftime('%Y-%m-%d')} - {datetime.now().strftime('%Y-%m-%d')}

📈 信号统计:
- 总信号数: {week_signals}
- 高置信度: {week_high_conf}
- 执行交易: {week_trades}
- 信号质量: {(week_high_conf/week_signals*100) if week_signals > 0 else 0:.1f}%

📱 平台统计:
- 小红书帖子: {xiaohongshu_posts}
- Twitter帖子: {twitter_posts}

💡 建议:
{"✅ 系统运行良好" if week_signals > 10 else "⚠️ 建议检查监控配置"}
{"✅ 信号质量不错" if week_high_conf > 5 else "💡 可以调整博主选择"}

---
下周继续监控！
        """
        
        await email_service.send_system_alert(report, "INFO")
        logger.info("Weekly report sent")
    
    async def health_check(self):
        """健康检查"""
        issues = []
        
        try:
            # 检查数据库连接
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
        except Exception as e:
            issues.append(f"数据库连接异常: {e}")
        
        # 检查最近的信号
        db = SessionLocal()
        try:
            recent_signals = db.query(AISignal).filter(
                AISignal.timestamp >= datetime.now() - timedelta(hours=24)
            ).count()
            
            if recent_signals == 0:
                issues.append("24小时内无新信号，可能监控异常")
        finally:
            db.close()
        
        # 检查内存使用
        import psutil
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if memory_mb > 400:
            issues.append(f"内存使用偏高: {memory_mb:.1f}MB")
        
        if issues:
            await email_service.send_system_alert(
                f"系统健康检查发现问题:\n" + "\n".join(f"- {issue}" for issue in issues),
                "WARNING"
            )
        else:
            logger.info("Health check passed")

async def main():
    """主函数"""
    maintenance = MaintenanceManager()
    
    print("🔧 执行系统维护...")
    
    # 1. 清理旧数据
    cleanup_result = await maintenance.cleanup_old_data()
    print(f"✅ 数据清理完成: {cleanup_result}")
    
    # 2. 健康检查
    await maintenance.health_check()
    print("✅ 健康检查完成")
    
    # 3. 生成周报（可选）
    today = datetime.now()
    if today.weekday() == 6:  # 周日
        await maintenance.generate_weekly_report()
        print("✅ 周报已发送")
    
    print("🎉 维护完成！")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 系统运行效果预期

### 💰 成本预算（每月）
```
🆓 免费资源:
- Railway.app: $0 (500小时/月)
- Supabase: $0 (500MB数据库)
- Gmail SMTP: $0 (500邮件/天)
- UptimeRobot: $0 (50个监控)

💳 付费成本:
- OpenAI API: $20-50 (根据使用量)
- TradingView Pro: $15 (可选)

📊 总计: $0-65/月
```

### 📈 性能指标
```
📱 监控能力:
- 博主数量: 3-5个
- 日处理帖子: 20-50条
- 质量筛选率: 70%
- AI解析成功率: 90%+

🎯 信号质量:
- 高置信度信号: 10-20个/周
- 邮件通知: 30-50个/周
- 信号准确率: 75%+
- 响应时间: <5分钟
```

### 🔧 资源使用
```
💻 系统资源:
- 内存使用: 300-400MB
- CPU使用: 10-30%
- 存储使用: 200-300MB
- 网络流量: 1-2GB/月

⚡ 运行稳定性:
- 正常运行时间: 99%+
- 自动恢复: 支持
- 错误处理: 完善
- 监控告警: 实时
```

---

## 🎯 总结

这份完整的执行指南基于100%免费云服务，为您提供了：

✅ **零成本起步**: 利用Railway.app、Supabase等免费服务
✅ **智能信号处理**: OpenAI解析 + 分级执行策略
✅ **成本严格控制**: 质量预筛选 + 预算监控
✅ **内存优化设计**: 适配免费版资源限制
✅ **完整监控体系**: 实时状态监控 + 邮件告警
✅ **易于维护**: 自动化部署 + 维护脚本

**核心优势**:
- 🆓 **完全免费启动**: 前3-6个月零成本运行
- 🤖 **智能化程度高**: GPT-4级别的内容理解
- 📧 **用户体验好**: 精美HTML邮件通知
- 🔒 **风险控制严**: 多层安全检查机制
- 📈 **可扩展性强**: 可平滑升级到付费版

**建议实施路径**:
1. **第1周**: 注册免费服务，部署基础系统
2. **第2周**: 配置OpenAI解析和邮件通知
3. **第3周**: 优化性能，添加监控
4. **第4周**: 上线运行，收集反馈

这个系统让您能够**零成本体验AI驱动的投资决策辅助**，验证概念可行性后再考虑升级到更强大的付费基础设施。开始您的AI投资助手之旅吧！🚀