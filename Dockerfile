# Dockerfile ultra-ligero para GPU en Fly.io
FROM ubuntu:22.04

# Variables de entorno para optimización
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface"
ENV HF_HOME="/root/.cache/huggingface"

# Instalar solo lo esencial
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /app

# Instalar todas las dependencias en una sola línea
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    python3.11 -m pip install --no-cache-dir diffusers==0.21.4 transformers==4.30.2 accelerate==0.20.3 && \
    python3.11 -m pip install --no-cache-dir huggingface_hub==0.16.4 && \
    python3.11 -m pip install --no-cache-dir gradio==3.50.2 && \
    python3.11 -m pip install --no-cache-dir Pillow==9.5.0 numpy==1.24.3 && \
    python3.11 -m pip install --no-cache-dir boto3==1.28.55 && \
    python3.11 -m pip install --no-cache-dir fastapi==0.104.1 uvicorn==0.24.0 && \
    python3.11 -m pip cache purge

# Verificar todas las dependencias instaladas
RUN python3.11 -m pip list | grep -E "(gradio|torch|diffusers|huggingface|transformers|accelerate|Pillow|numpy|boto3|fastapi|uvicorn)"

# Copiar el código de la aplicación
COPY app.py .

# Crear directorio para cache de modelos
RUN mkdir -p /root/.cache/huggingface && \
    chmod 755 /root/.cache/huggingface

# Limpiar todo el cache
RUN rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Exponer puerto
EXPOSE 7860

# Comando de inicio optimizado usando python3.11 explícitamente
CMD ["python3.11", "app.py"] 