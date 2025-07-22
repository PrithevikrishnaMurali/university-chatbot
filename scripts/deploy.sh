#!/bin/bash
set -e

echo "🚀 Starting University Chatbot deployment..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

mkdir -p data logs

if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.production .env
    echo "⚠️  Please update the .env file with your configurations"
fi

echo "🔧 Building and starting services..."
docker-compose up -d --build

echo "⏳ Waiting for services to start..."
sleep 10

if docker-compose ps | grep -q "university_chatbot_api.*Up"; then
    echo "✅ API service is running"
else
    echo "❌ API service failed to start"
    docker-compose logs chatbot_api
    exit 1
fi

if docker-compose ps | grep -q "university_chatbot_mongodb.*Up"; then
    echo "✅ MongoDB service is running"
else
    echo "❌ MongoDB service failed to start"
    docker-compose logs mongodb
    exit 1
fi

echo "📊 Loading sample data..."
sleep 5
docker-compose exec chatbot_api python load_data.py

echo "🎉 Deployment completed successfully!"
echo "📡 API is available at: http://localhost:8000"
echo "🗄️  MongoDB Express is available at: http://localhost:8081"
echo "📖 API Documentation: http://localhost:8000/docs"