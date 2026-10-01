FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_ROOT=/models \
    TESTRX_CONFIG=/app/config.yaml \
    TESTRX_LOG_DB=/data/testrx.sqlite3 \
    TESTRX_BUILD_REPORT=/data/index-build-report.json

WORKDIR /app
COPY pyproject.toml ./
COPY requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps .

COPY config.yaml ./config.yaml
EXPOSE 8000
CMD ["uvicorn", "testrx_prod.application:api", "--host", "0.0.0.0", "--port", "8000"]
