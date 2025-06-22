#!/bin/bash

# Build script for creating Ubuntu 20.04 compatible executable

echo "🚀 Building Ubuntu 20.04 compatible executable..."

# Build the Docker image
echo "📦 Building Docker image with Ubuntu 20.04..."
if ! docker build -f Dockerfile.ubuntu -t blurapi-builder-ubuntu .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating executable..."
if ! docker create --name blurapi-container-ubuntu blurapi-builder-ubuntu; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-ubuntu 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting executable..."
if ! docker cp blurapi-container-ubuntu:/output/blurapi ./blurapi-ubuntu; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-ubuntu 2>/dev/null || true
    docker rmi blurapi-builder-ubuntu 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-ubuntu 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-ubuntu

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-ubuntu"
echo ""
echo "To use on Ubuntu/Debian/CentOS:"
echo "1. Copy blurapi-ubuntu to your Linux server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-ubuntu --help" 