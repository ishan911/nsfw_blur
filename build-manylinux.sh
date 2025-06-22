#!/bin/bash

# Build script for creating manylinux2014 compatible executable

echo "🚀 Building manylinux2014 compatible executable..."

# Build the Docker image
echo "📦 Building Docker image with manylinux2014..."
if ! docker build -f Dockerfile.manylinux -t blurapi-builder-manylinux .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating executable..."
if ! docker create --name blurapi-container-manylinux blurapi-builder-manylinux; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-manylinux 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting executable..."
if ! docker cp blurapi-container-manylinux:/output/blurapi ./blurapi-manylinux; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-manylinux 2>/dev/null || true
    docker rmi blurapi-builder-manylinux 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-manylinux 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-manylinux

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-manylinux"
echo ""
echo "To use on any Linux distribution (CentOS, Ubuntu, etc.):"
echo "1. Copy blurapi-manylinux to your Linux server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-manylinux --help"
echo ""
echo "This executable is compatible with:"
echo "- CentOS 6+"
echo "- Ubuntu 14.04+"
echo "- Debian 8+"
echo "- And many other Linux distributions" 