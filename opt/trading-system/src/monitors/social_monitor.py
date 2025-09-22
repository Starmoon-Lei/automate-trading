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