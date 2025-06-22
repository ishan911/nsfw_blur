#!/bin/bash

# Build script for creating CentOS-compatible executable

echo "🚀 Building CentOS-compatible executable..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t blurapi-builder .

# Create a container and copy the executable
echo "🔧 Creating executable..."
docker create --name blurapi-container blurapi-builder

# Copy the executable from the container
echo "📋 Extracting executable..."
docker cp blurapi-container:/output/blurapi ./blurapi-centos

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container

# Make the executable executable
chmod +x ./blurapi-centos

echo "✅ Build complete!"
echo "📁 Executable created: ./blurapi-centos"
echo ""
echo "To use on CentOS:"
echo "1. Copy blurapi-centos to your CentOS server"
echo "2. Make sure you have the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-centos --help" 