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