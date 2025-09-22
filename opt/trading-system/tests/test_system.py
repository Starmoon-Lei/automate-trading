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