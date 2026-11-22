# ---- build stage: wheels ----
FROM python:3.12-slim AS build
WORKDIR /w
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime ----
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
WORKDIR /app
COPY . .
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app
EXPOSE 8000 8501
# default: API. docker-compose overrides the command for the UI.
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
