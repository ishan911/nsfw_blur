#!/bin/bash

# Test script to verify Docker is working properly

echo "🔍 Testing Docker setup..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Test pulling Rocky Linux image
echo "📦 Testing Rocky Linux 8 image pull..."
if ! docker pull rockylinux:8; then
    echo "❌ Failed to pull Rocky Linux 8 image!"
    echo "This might be a network issue or Docker registry problem."
    exit 1
fi

echo "✅ Rocky Linux 8 image pulled successfully"

# Test pulling Ubuntu image
echo "📦 Testing Ubuntu 20.04 image pull..."
if ! docker pull ubuntu:20.04; then
    echo "❌ Failed to pull Ubuntu 20.04 image!"
    echo "This might be a network issue or Docker registry problem."
    exit 1
fi

echo "✅ Ubuntu 20.04 image pulled successfully"

echo ""
echo "🎉 Docker test completed successfully!"
echo "You can now run the build scripts:"
echo "  ./build-rocky.sh    # For Rocky Linux 8 (recommended)"
echo "  ./build-ubuntu.sh   # For Ubuntu 20.04 (alternative)" 