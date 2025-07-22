#!/bin/bash

# University Chatbot Deployment Script
# This script helps deploy your chatbot to various platforms

set -e  # Exit on any error

echo "🚀 University Chatbot Deployment Script"
echo "======================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create a .env file with your MongoDB credentials."
    echo "Use .env.example as a template."
    exit 1
fi

# Load environment variables
source .env

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Heroku
deploy_heroku() {
    echo "🟣 Deploying to Heroku..."
    
    if ! command_exists heroku; then
        echo "❌ Heroku CLI not found. Please install it first."
        return 1
    fi
    
    echo "📝 Creating Heroku app..."
    read -p "Enter your Heroku app name: " app_name
    
    heroku create "$app_name" || echo "App might already exist, continuing..."
    
    echo "🔧 Setting environment variables..."
    heroku config:set MONGODB_URL="$MONGODB_URL" --app "$app_name"
    heroku config:set DATABASE_NAME="$DATABASE_NAME" --app "$app_name"
    heroku config:set DEBUG=False --app "$app_name"
    
    echo "🚀 Deploying to Heroku..."
    git push heroku main
    
    echo "🌐 Opening app..."
    heroku open --app "$app_name"
}

# Function to deploy to Railway
deploy_railway() {
    echo "🚄 Deploying to Railway..."
    
    if ! command_exists railway; then
        echo "❌ Railway CLI not found. Installing..."
        npm install -g @railway/cli
    fi
    
    echo "🔧 Initializing Railway project..."
    railway login
    railway init
    
    echo "📝 Setting environment variables..."
    railway variables set MONGODB_URL="$MONGODB_URL"
    railway variables set DATABASE_NAME="$DATABASE_NAME"
    railway variables set DEBUG=False
    
    echo "🚀 Deploying to Railway..."
    railway up
}

# Function to build Docker image
build_docker() {
    echo "🐳 Building Docker image..."
    
    if ! command_exists docker; then
        echo "❌ Docker not found. Please install Docker first."
        return 1
    fi
    
    echo "📦 Building image..."
    docker build -t university-chatbot .
    
    echo "✅ Docker image built successfully!"
    echo "🏃 To run locally: docker run -p 8000:8000 --env-file .env university-chatbot"
}

# Function to prepare for Vercel
prepare_vercel() {
    echo "▲ Preparing for Vercel deployment..."
    
    if ! command_exists vercel; then
        echo "❌ Vercel CLI not found. Installing..."
        npm install -g vercel
    fi
    
    echo "🔧 Setting up Vercel project..."
    vercel login
    
    echo "📝 Please set these environment variables in Vercel dashboard:"
    echo "   MONGODB_URL: $MONGODB_URL"
    echo "   DATABASE_NAME: $DATABASE_NAME"
    echo "   DEBUG: False"
    echo ""
    echo "🌐 Then run: vercel --prod"
}

# Function to prepare for Render
prepare_render() {
    echo "🎨 Preparing for Render deployment..."
    echo ""
    echo "📋 Steps to deploy on Render:"
    echo "1. Go to https://render.com"
    echo "2. Connect your GitHub repository"
    echo "3. Create a new Web Service"
    echo "4. Set these environment variables:"
    echo "   MONGODB_URL: $MONGODB_URL"
    echo "   DATABASE_NAME: $DATABASE_NAME"
    echo "   DEBUG: False"
    echo "   PORT: 8000"
    echo "5. Set build command: pip install -r requirements.txt"
    echo "6. Set start command: python main.py"
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    
    echo "📡 Testing database connection..."
    python data_validator.py
    
    echo "🔧 Testing backend functionality..."
    python debug_backend.py
    
    echo "✅ Tests completed!"
}

# Function to start local development
start_local() {
    echo "💻 Starting local development server..."
    
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    
    echo "🧪 Running tests first..."
    run_tests
    
    echo "🚀 Starting server..."
    python main.py
}

# Main menu
echo ""
echo "Choose deployment option:"
echo "1) Start local development"
echo "2) Run tests only"
echo "3) Build Docker image"
echo "4) Deploy to Heroku"
echo "5) Deploy to Railway"
echo "6) Prepare for Vercel"
echo "7) Prepare for Render"
echo "8) Exit"
echo ""

read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        start_local
        ;;
    2)
        run_tests
        ;;
    3)
        build_docker
        ;;
    4)
        deploy_heroku
        ;;
    5)
        deploy_railway
        ;;
    6)
        prepare_vercel
        ;;
    7)
        prepare_render
        ;;
    8)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Deployment script completed!"
echo "🌐 Don't forget to update your frontend API URL if deploying to a new domain."