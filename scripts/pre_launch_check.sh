echo "🔍 Pre-launch System Check"
echo "=========================="

# 1. 环境变量检查
echo "1. 检查环境变量..."
check_env() {
    if [ -z "${!1}" ]; then
        echo "❌ $1 未设置"
        return 1
    else
        echo "✅ $1 已设置"
        return 0
    fi
}

check_env "OPENAI_API_KEY"
check_env "DATABASE_URL"
check_env "SMTP_USERNAME"
check_env "SMTP_PASSWORD"
check_env "ALERT_EMAIL"

# 2. 数据库连接检查
echo "2. 检查数据库连接..."
python -c "
from src.models.database import SessionLocal
try:
    db = SessionLocal()
    db.execute('SELECT 1')
    db.close()
    print('✅ 数据库连接正常')
except Exception as e:
    print(f'❌ 数据库连接失败: {e}')
    exit(1)
"

# 3. OpenAI API检查
echo "3. 检查OpenAI API..."
python -c "
import openai
from src.config.settings import settings
try:
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=5
    )
    print('✅ OpenAI API连接正常')
except Exception as e:
    print(f'❌ OpenAI API连接失败: {e}')
    exit(1)
"

# 4. 邮件服务检查
echo "4. 检查邮件服务..."
python -c "
import asyncio
from src.utils.email_service import email_service

async def test_email():
    try:
        success = await email_service.send_system_alert('系统测试邮件', 'INFO')
        if success:
            print('✅ 邮件服务正常')
        else:
            print('❌ 邮件发送失败')
    except Exception as e:
        print(f'❌ 邮件服务错误: {e}')

asyncio.run(test_email())
"

echo "5. 系统资源检查..."
python -c "
import psutil
memory_mb = psutil.virtual_memory().available / 1024 / 1024
cpu_count = psutil.cpu_count()
print(f'✅ 可用内存: {memory_mb:.0f}MB')
print(f'✅ CPU核心数: {cpu_count}')
"

echo ""
echo "🚀 系统检查完成！"
echo "📧 请检查邮箱是否收到测试邮件"
echo "🌐 系统准备就绪，可以启动服务"