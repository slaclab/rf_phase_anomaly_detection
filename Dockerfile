FROM python:3.12-slim

WORKDIR /app
# Set env variable for k2eg
ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config
COPY requirements.txt .

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy python3-dev g++ gcc && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -qy python3-dev g++ gcc --purge && \
    rm -rf /var/lib/apt/lists/*

COPY . .