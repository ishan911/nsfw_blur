#!/bin/bash

# Build script for creating a completely standalone binary

echo "🚀 Building standalone binary (no Python dependencies required)..."

# Check if model exists
if [ ! -f "models/640m.onnx" ]; then
    echo "❌ Model not found: models/640m.onnx"
    echo "Please make sure your ONNX model is in the models/ directory."
    exit 1
fi

# Build the Docker image
echo "📦 Building Docker image with standalone binary support..."
if ! docker build -f Dockerfile.standalone -t blurapi-builder-standalone .; then
    echo "❌ Docker build failed!"
    echo "Please check the error messages above and try again."
    exit 1
fi

# Create a container and copy the executable
echo "🔧 Creating standalone executable..."
if ! docker create --name blurapi-container-standalone blurapi-builder-standalone; then
    echo "❌ Failed to create container!"
    echo "Cleaning up..."
    docker rmi blurapi-builder-standalone 2>/dev/null || true
    exit 1
fi

# Copy the executable from the container
echo "📋 Extracting standalone executable..."
if ! docker cp blurapi-container-standalone:/output/blurapi-standalone ./blurapi-standalone; then
    echo "❌ Failed to extract executable!"
    echo "Cleaning up..."
    docker rm blurapi-container-standalone 2>/dev/null || true
    docker rmi blurapi-builder-standalone 2>/dev/null || true
    exit 1
fi

# Clean up the container
echo "🧹 Cleaning up..."
docker rm blurapi-container-standalone 2>/dev/null || true

# Make the executable executable
chmod +x ./blurapi-standalone

# Show file size
file_size=$(du -h ./blurapi-standalone | cut -f1)
echo "📊 Executable size: $file_size"

echo "✅ Standalone binary build complete!"
echo "📁 Executable created: ./blurapi-standalone"
echo ""
echo "🎉 This is a TRUE BINARY with:"
echo "✅ No Python installation required"
echo "✅ No dependencies to install"
echo "✅ Self-contained (includes Python runtime)"
echo "✅ Works on any compatible Linux system"
echo ""
echo "To use:"
echo "1. Copy blurapi-standalone to your Linux server"
echo "2. Copy the models/ directory with your ONNX model"
echo "3. Run: ./blurapi-standalone --help"
echo ""
echo "Note: The executable will be larger (~100-200MB) because it includes"
echo "the entire Python runtime and all dependencies." 