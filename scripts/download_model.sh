#!/bin/bash

# Download script for Phi-3-mini-4k-instruct GGUF model
# This is a small, efficient model perfect for 8GB memory systems

set -e

MODEL_DIR="./models"
MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
MODEL_FILE="phi-3-mini-4k-instruct-q4.gguf"

echo "🚀 Downloading Phi-3-mini-4k-instruct GGUF model..."
echo "📁 Model directory: $MODEL_DIR"
echo "🔗 Download URL: $MODEL_URL"

# Create models directory if it doesn't exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "📁 Creating models directory..."
    mkdir -p "$MODEL_DIR"
fi

# Check if model already exists
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_DIR/$MODEL_FILE"
    echo "🔄 To re-download, delete the existing file first"
    exit 0
fi

# Download the model
echo "⬇️  Downloading model..."
if command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
else
    echo "❌ Error: Neither wget nor curl is available"
    echo "Please install one of these tools to download the model"
    exit 1
fi

# Verify download
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✅ Model downloaded successfully!"
    echo "📊 File size: $(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1)"
    echo "📝 Model path: $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Set MODEL_PATH environment variable to: $MODEL_DIR/$MODEL_FILE"
    echo "2. Start the server: python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000"
else
    echo "❌ Error: Download failed"
    exit 1
fi
