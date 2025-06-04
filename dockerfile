# Multi-stage Docker build for Parking Space Detection System
# Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    # Video and camera support
    libv4l-dev \
    v4l-utils \
    # GUI support (for real-time display)
    libx11-6 \
    libxcb1 \
    libxau6 \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Build tools (for compiling packages)
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Utilities
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Production stage
FROM base as production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Clean up pip cache
    rm -rf ~/.cache/pip

# Copy application code
COPY parking_detection_system.py .

# Create directories for data and output
RUN mkdir -p /app/dataset /app/output /app/models && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set default command
ENTRYPOINT ["python", "parking_detection_system.py"]
CMD ["--help"]

# Development stage (includes additional tools)
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    # X11 forwarding for GUI
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies including development packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Additional development packages
    pip install \
    jupyter \
    pytest \
    pytest-cov \
    flake8 \
    black \
    isort \
    ipykernel \
    && rm -rf ~/.cache/pip

# Copy application code
COPY . .

# Create directories
RUN mkdir -p dataset output models

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["bash"]

# GPU-enabled stage (for future deep learning features)
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    # OpenCV and GUI dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libv4l-dev \
    v4l-utils \
    libx11-6 \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Build tools
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Install GPU-accelerated packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install tensorflow-gpu && \
    rm -rf ~/.cache/pip

# Copy application
COPY parking_detection_system.py .

# Create directories
RUN mkdir -p dataset output models

ENTRYPOINT ["python", "parking_detection_system.py"]

# Docker Compose services configuration
# Create a docker-compose.yml file separately for orchestration

# Build arguments for customization
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=latest

# Metadata
LABEL maintainer="your-email@example.com" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="parking-space-detector" \
      org.label-schema.description="Automated Parking Space Detection System" \
      org.label-schema.url="https://github.com/your-repo/parking-detection-system" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-repo/parking-detection-system" \
      org.label-schema.vendor="Your Organization" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2, sklearn, numpy; print('Dependencies OK')" || exit 1

# Volume mounts for data persistence
VOLUME ["/app/dataset", "/app/output", "/app/models"]

# Default environment variables (can be overridden)
ENV MODEL_PATH=/app/models/parking_model.pkl \
    OUTPUT_PATH=/app/output \
    DATASET_PATH=/app/dataset \
    CAMERA_ID=0 \
    LOG_LEVEL=INFO

# Security: Run as non-root user in production
USER appuser
