#!/bin/bash

# Build script for creating Rocky Linux 8 compatible executable (CentOS alternative)

echo "🚀 Building Rocky Linux 8 compatible executable..."

# Build the Docker image
echo "📦 Building Docker image with Rocky Linux 8..."
if ! docker build -t blurapi-builder-rocky .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating executable..."
if ! docker create --name blurapi-container-rocky blurapi-builder-rocky; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-rocky 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting executable..."
if ! docker cp blurapi-container-rocky:/output/blurapi ./blurapi-rocky; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-rocky 2>/dev/null || true
    docker rmi blurapi-builder-rocky 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-rocky 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-rocky

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-rocky"
echo ""
echo "To use on CentOS/Rocky Linux:"
echo "1. Copy blurapi-rocky to your CentOS/Rocky Linux server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-rocky --help" 