#!/bin/bash
# deploy.sh

echo "🚀 Deploying Free AI Trading System..."

# 1. 检查必需的环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    exit 1
fi

if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL not set"
    exit 1
fi

# 2. 安装Railway CLI (如果没有)
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# 3. 登录Railway
echo "🔐 Login to Railway..."
railway login

# 4. 初始化项目 (如果是第一次)
if [ ! -f "railway.json" ]; then
    echo "🆕 Initializing Railway project..."
    railway init
fi

# 5. 设置环境变量
echo "⚙️ Setting environment variables..."
railway variables set OPENAI_API_KEY="$OPENAI_API_KEY"
railway variables set DATABASE_URL="$DATABASE_URL"
railway variables set SMTP_USERNAME="$SMTP_USERNAME"
railway variables set SMTP_PASSWORD="$SMTP_PASSWORD"
railway variables set ALERT_EMAIL="$ALERT_EMAIL"

# 6. 部署
echo "🚀 Deploying to Railway..."
railway deploy

# 7. 获取部署URL
echo "🌐 Getting deployment URL..."
railway status

echo "✅ Deployment completed!"
echo "📊 Check your system at: https://your-app.railway.app"
echo "📧 Check your email for startup notification"