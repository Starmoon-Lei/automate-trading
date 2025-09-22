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