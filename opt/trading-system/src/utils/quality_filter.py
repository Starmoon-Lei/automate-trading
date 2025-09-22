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