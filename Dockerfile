FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Dashboard port
EXPOSE 8090

CMD ["python", "-m", "src.core"]
