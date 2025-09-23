# src/main.py
import asyncio
import logging
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import signal
import gc
import psutil

from src.trading_engine import trading_engine
from src.models.database import db_manager
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

# 全局变量
scheduler_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global scheduler_task

    # 启动逻辑
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

    yield

    # 关闭逻辑
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

# 创建FastAPI应用
app = FastAPI(
    lifespan=lifespan,
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
        await db_manager.execute_query("SELECT 1")
        
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
        # 今日统计
        today = datetime.now().date()

        # 使用异步查询获取今日信号数量
        today_signals_query = """
            SELECT COUNT(*) FROM ai_signals
            WHERE timestamp::date = $1
        """
        today_signals_result = await db_manager.execute_query(today_signals_query, today)
        today_signals = today_signals_result['count'] if today_signals_result else 0

        # 高置信度信号数量
        today_high_conf_query = """
            SELECT COUNT(*) FROM ai_signals
            WHERE timestamp::date = $1 AND confidence >= $2
        """
        today_high_conf_result = await db_manager.execute_query(
            today_high_conf_query, today, settings.HIGH_CONFIDENCE_THRESHOLD
        )
        today_high_conf = today_high_conf_result['count'] if today_high_conf_result else 0

        # 今日交易数量
        today_trades_query = """
            SELECT COUNT(*) FROM trades
            WHERE timestamp::date = $1
        """
        today_trades_result = await db_manager.execute_query(today_trades_query, today)
        today_trades = today_trades_result['count'] if today_trades_result else 0

        # AI使用统计
        ai_stats = trading_engine.ai_parser.get_usage_stats()

        # 内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
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
        # 使用异步查询获取最近的信号
        signals_query = """
            SELECT id, symbol, action, confidence, target_price, reasoning,
                   risk_level, status, timestamp
            FROM ai_signals
            ORDER BY timestamp DESC
            LIMIT $1
        """

        signals_result = await db_manager.fetch_all(signals_query, limit)

        signals_list = []
        if signals_result:
            for signal in signals_result:
                reasoning = signal['reasoning']
                if reasoning and len(reasoning) > 100:
                    reasoning = reasoning[:100] + "..."

                signals_list.append({
                    "id": signal['id'],
                    "symbol": signal['symbol'],
                    "action": signal['action'],
                    "confidence": signal['confidence'],
                    "target_price": signal['target_price'],
                    "reasoning": reasoning,
                    "risk_level": signal['risk_level'],
                    "status": signal['status'],
                    "timestamp": signal['timestamp'].isoformat() if signal['timestamp'] else None
                })

        return {
            "signals": signals_list,
            "total": len(signals_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {e}")

@app.get("/trades")
async def get_recent_trades(limit: int = 10):
    """获取最近的交易"""
    try:
        # 使用异步查询获取最近的交易
        trades_query = """
            SELECT id, symbol, action, quantity, entry_price, status, timestamp
            FROM trades
            ORDER BY timestamp DESC
            LIMIT $1
        """

        trades_result = await db_manager.fetch_all(trades_query, limit)

        trades_list = []
        if trades_result:
            for trade in trades_result:
                trades_list.append({
                    "id": trade['id'],
                    "symbol": trade['symbol'],
                    "action": trade['action'],
                    "quantity": trade['quantity'],
                    "entry_price": trade['entry_price'],
                    "status": trade['status'],
                    "timestamp": trade['timestamp'].isoformat() if trade['timestamp'] else None
                })

        return {
            "trades": trades_list,
            "total": len(trades_list)
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