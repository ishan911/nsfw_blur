#!/bin/bash

# Build script for creating Rocky Linux 8 compatible executable (simplified version)

echo "🚀 Building Rocky Linux 8 compatible executable (simplified)..."

# Build the Docker image
echo "📦 Building Docker image with Rocky Linux 8 (simplified)..."
if ! docker build -f Dockerfile.simple -t blurapi-builder-simple .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating executable..."
if ! docker create --name blurapi-container-simple blurapi-builder-simple; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-simple 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting executable..."
if ! docker cp blurapi-container-simple:/output/blurapi ./blurapi-simple; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-simple 2>/dev/null || true
    docker rmi blurapi-builder-simple 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-simple 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-simple

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-simple"
echo ""
echo "To use on CentOS/Rocky Linux:"
echo "1. Copy blurapi-simple to your CentOS/Rocky Linux server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-simple --help" 