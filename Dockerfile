# 期货量化交易系统 v2.0
FROM python:3.11-slim

LABEL maintainer="atxinsky"
LABEL version="2.0"
LABEL description="Futures Quantitative Trading System with Backtest & Live Trading"

WORKDIR /app

# 使用阿里云镜像源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p /app/data /app/logs /app/cache /app/backup

# 暴露端口
EXPOSE 8502

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# 启动命令 - 使用新的 app/main.py
CMD ["streamlit", "run", "app/main.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]
