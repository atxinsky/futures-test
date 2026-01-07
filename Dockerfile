# 期货量化交易系统 v2.0
FROM python:3.11-slim

LABEL maintainer="tretra"
LABEL version="2.0"
LABEL description="Futures Quantitative Trading System with TqSdk Live Trading"

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

# 安装Python依赖 (包含tqsdk)
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p /app/data /app/logs /app/cache /app/backup

# 设置时区
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8504

# 健康检查 (Web模式)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8504/_stcore/health || exit 1

# 默认启动命令 - Web界面
CMD ["streamlit", "run", "app/main.py", "--server.port=8504", "--server.address=0.0.0.0", "--server.headless=true"]
