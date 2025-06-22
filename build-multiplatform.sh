#!/bin/bash

# Build script for creating multi-platform compatible executable

echo "🚀 Building multi-platform compatible executable..."

# Build the Docker image
echo "📦 Building Docker image with multi-platform support..."
if ! docker build -f Dockerfile.multiplatform -t blurapi-builder-multiplatform .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating executable..."
if ! docker create --name blurapi-container-multiplatform blurapi-builder-multiplatform; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-multiplatform 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting executable..."
if ! docker cp blurapi-container-multiplatform:/output/blurapi ./blurapi-multiplatform; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-multiplatform 2>/dev/null || true
    docker rmi blurapi-builder-multiplatform 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-multiplatform 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-multiplatform

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-multiplatform"
echo ""
echo "This executable is compatible with:"
echo "- AMD64 Linux distributions (CentOS, Ubuntu, etc.)"
echo "- Built on ARM64 Mac but targets AMD64 Linux"
echo "- Maximum compatibility with Linux servers"
echo ""
echo "To use:"
echo "1. Copy blurapi-multiplatform to your Linux server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-multiplatform --help" 