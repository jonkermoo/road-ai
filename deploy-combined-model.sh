#!/bin/bash

# Deploy combined YOLO model to EC2
# This script uploads the trained model weights and deploys the updated backend

set -e

# Configuration
EC2_USER="ubuntu"
EC2_HOST="roadai.online"
LOCAL_MODEL="ml/runs/detect/combined_road_ai/weights/best.pt"
REMOTE_MODEL_DIR="~/road-ai/ml/runs/detect/combined_road_ai/weights"

echo "🚀 Deploying Combined YOLO Model to EC2..."
echo ""

# Check if model exists locally
if [ ! -f "$LOCAL_MODEL" ]; then
    echo "❌ Error: Model not found at $LOCAL_MODEL"
    echo "   Make sure you've trained the model first with: python ml/train_combined.py"
    exit 1
fi

# Get model file size
MODEL_SIZE=$(du -h "$LOCAL_MODEL" | cut -f1)
echo "📦 Model size: $MODEL_SIZE"
echo ""

# Create remote directory structure
echo "📁 Creating remote directory structure..."
ssh ${EC2_USER}@${EC2_HOST} "mkdir -p ${REMOTE_MODEL_DIR}"

# Upload model weights
echo "📤 Uploading model weights to EC2..."
echo "   This may take a few minutes depending on your connection..."
scp "$LOCAL_MODEL" ${EC2_USER}@${EC2_HOST}:${REMOTE_MODEL_DIR}/best.pt

echo ""
echo "✅ Model uploaded successfully!"
echo ""

# Pull latest code and restart service
echo "📥 Pulling latest code from GitHub..."
ssh ${EC2_USER}@${EC2_HOST} "cd ~/road-ai && git pull origin main"

echo ""
echo "🔄 Restarting backend service..."
ssh ${EC2_USER}@${EC2_HOST} "sudo systemctl restart road-ai-backend"

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 3

# Check service status
echo ""
echo "✅ Checking service status..."
ssh ${EC2_USER}@${EC2_HOST} "sudo systemctl status road-ai-backend --no-pager -l | head -20"

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. Check the video feed at https://roadai.online"
echo "  2. Monitor logs: ssh ${EC2_USER}@${EC2_HOST} 'sudo journalctl -u road-ai-backend -f'"
echo "  3. Expected performance: 10-15 FPS (3x faster than before!)"
echo ""
