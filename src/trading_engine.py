# src/trading_engine.py
import asyncio
import logging
import gc
from typing import List, Dict, Any
from datetime import datetime
from src.parsers.openai_parser import OptimizedOpenAIParser
from src.monitors.social_monitor import LightweightSocialMonitor
from src.utils.email_service import email_service
from src.models.database import db_manager
from src.config.settings import settings
import json
import psutil

class MemoryOptimizedTradingEngine:
    """内存优化的交易引擎 - 适配免费版限制"""
    
    def __init__(self):
        self.ai_parser = OptimizedOpenAIParser()
        self.db_manager = db_manager
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
                try:
                    # Validate signal has required attributes
                    if not hasattr(signal, 'confidence') or signal.confidence is None:
                        self.logger.warning(f"Signal missing confidence attribute: {signal}")
                        continue

                    if not hasattr(signal, 'symbol'):
                        self.logger.warning(f"Signal missing symbol attribute: {signal}")
                        continue

                    # Set default risk_level if missing
                    if not hasattr(signal, 'risk_level'):
                        signal.risk_level = "UNKNOWN"

                    # Now safe to access attributes
                    if (signal.confidence >= settings.HIGH_CONFIDENCE_THRESHOLD and
                        result.credibility_score >= settings.MIN_CREDIBILITY_SCORE):
                        high_conf_signals.append(signal)
                    elif signal.confidence >= settings.MEDIUM_CONFIDENCE_THRESHOLD:
                        low_conf_signals.append(signal)

                except AttributeError as e:
                    self.logger.error(f"Signal missing required attribute: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing signal: {e}")
                    continue
            
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