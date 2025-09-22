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