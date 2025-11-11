#!/bin/bash

# Simple deployment script for Road AI backend
# Usage: ./deploy-backend.sh

set -e  # Exit on error

echo "🚀 Deploying Road AI Backend to EC2..."

# Pull latest changes
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Restart the backend service
echo "🔄 Restarting backend service..."
sudo systemctl restart road-ai-backend

# Wait a moment for service to start
sleep 2

# Check service status
echo "✅ Checking service status..."
sudo systemctl status road-ai-backend --no-pager -l

echo "🎉 Deployment complete!"
