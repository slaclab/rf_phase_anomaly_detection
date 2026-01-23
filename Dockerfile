FROM python:3.12-slim

WORKDIR /app

# Set env variable for k2eg
ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config

# Install system dependencies, then clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy python3-dev g++ gcc git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove python3-dev g++ gcc && \
    rm -rf /root/.cache

COPY . .

CMD ["python", "-m", "main", "-ll", "20"]
