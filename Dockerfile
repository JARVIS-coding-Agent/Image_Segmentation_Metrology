# 動物眼睛量測 Pipeline — CPU 離線執行映像
FROM python:3.11-slim

# OpenCV / matplotlib 執行期所需系統庫
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml ./

# 預設顯示說明；實際執行由 docker compose run 覆寫指令
ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["--help"]
