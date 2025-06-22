# Use Python 3 image for the runtime container
FROM python:3

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Essential libraries for OpenCV and image processing
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        # Basic utilities
        wget \
        curl \
        git \
        # Build essentials
        gcc \
        g++ \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install latest ONNX Runtime for compatibility
RUN pip install onnxruntime==1.18.0

# Copy source code
COPY . .

# Create directories for data
RUN mkdir -p /app/data /app/wp-content/uploads/backup

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "main.py", "--help"] 