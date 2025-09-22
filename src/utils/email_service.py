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