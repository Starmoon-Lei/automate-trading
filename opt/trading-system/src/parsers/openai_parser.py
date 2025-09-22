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