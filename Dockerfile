FROM python:3.12-slim

WORKDIR /app

# Set env variable for k2eg
ENV K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/app/config

# Install system dependencies, then clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -qy python3-dev g++ gcc git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Torch is a lume-model dependency. Here we are installing the CPU version to save on image size.
# If you need GPU support, change the index-url to the appropriate one for your CUDA version.
RUN pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove python3-dev g++ gcc && \
    rm -rf /root/.cache

COPY . .