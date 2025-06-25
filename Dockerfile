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

# Crear enlace simbólico para python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias de Python con versiones compatibles y ligeras
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install --no-cache-dir diffusers==0.21.4 transformers==4.30.2 accelerate==0.20.3 && \
    pip3 install --no-cache-dir gradio==3.40.1 huggingface-hub==0.16.4 safetensors==0.3.1 && \
    pip3 install --no-cache-dir Pillow==9.5.0 numpy==1.24.3 && \
    pip3 cache purge

# Debug: mostrar info de gradio
RUN python3 -m pip show gradio

# Copiar el código de la aplicación
COPY app.py .

# Crear directorio para cache de modelos
RUN mkdir -p /root/.cache/huggingface && \
    chmod 755 /root/.cache/huggingface

# Limpiar todo el cache
RUN rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Exponer puerto
EXPOSE 7860

# Comando de inicio optimizado
CMD ["python3", "app.py"] 