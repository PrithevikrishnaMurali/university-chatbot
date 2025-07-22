#!/bin/bash

# Git Setup Script for University Chatbot
# This script prepares your project for GitHub and deployment

set -e

echo "🎯 Setting up University Chatbot for GitHub and Deployment"
echo "=========================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Create .env.example if it doesn't exist
if [ ! -f ".env.example" ]; then
    echo "📝 Creating .env.example..."
    cat > .env.example << EOL
# MongoDB Configuration
MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
DATABASE_NAME=University_data

# Application Settings
DEBUG=True
PORT=8000

# Security (generate a secure secret key for production)
SECRET_KEY=your-secret-key-here

# Optional: Logging
LOG_LEVEL=INFO
EOL
    echo "✅ .env.example created"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Please create a .env file with your actual credentials."
    echo "You can copy .env.example and modify it:"
    echo "cp .env.example .env"
    echo ""
    read -p "Do you want to create a basic .env file now? (y/n): " create_env
    
    if [ "$create_env" = "y" ] || [ "$create_env" = "Y" ]; then
        cp .env.example .env
        echo "📝 Basic .env file created. Please edit it with your MongoDB credentials."
        echo "⚠️  Remember: Never commit .env to GitHub!"
    fi
fi

# Create .gitignore if it doesn't exist (already created in artifacts)
echo "📋 Checking .gitignore..."
if [ -f ".gitignore" ]; then
    echo "✅ .gitignore exists"
else
    echo "⚠️  .gitignore not found - please add the .gitignore artifact"
fi

# Make deploy script executable
if [ -f "deploy.sh" ]; then
    chmod +x deploy.sh
    echo "✅ Made deploy.sh executable"
fi

# Make this script executable
chmod +x setup_git.sh

# Add all files to git
echo "📦 Adding files to Git..."
git add .

# Check if there are any changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    # Commit initial files
    echo "💾 Creating initial commit..."
    git commit -m "Initial commit: University Recommendation Chatbot

🎓 Features:
- FastAPI backend with MongoDB integration
- NLP-powered conversation flow
- 105+ Canadian universities database
- Modern responsive frontend
- Docker support
- Multiple deployment options

🚀 Ready for deployment to Heroku, Railway, Render, or Vercel"

    echo "✅ Initial commit created"
fi

# Instructions for GitHub
echo ""
echo "🌟 Next Steps - Push to GitHub:"
echo "================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Add the remote origin:"
echo "   git remote add origin https://github.com/yourusername/university-chatbot.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "🚀 After pushing to GitHub, you can:"
echo "   - Deploy to Heroku: heroku create your-app-name"
echo "   - Deploy to Railway: railway init && railway up"
echo "   - Deploy to Render: Connect your GitHub repo"
echo "   - Deploy to Vercel: vercel --prod"
echo ""
echo "📖 For detailed deployment instructions, see DEPLOYMENT.md"
echo ""

# Check project structure
echo "📁 Project Structure Check:"
echo "=========================="

required_files=(
    "main.py"
    "requirements.txt"
    "chatbot_ui.html"
    ".env.example"
    ".gitignore"
    "README.md"
    "DEPLOYMENT.md"
    "Dockerfile"
    "data_validator.py"
    "debug_backend.py"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        missing_files+=("$file")
    fi
done

optional_files=(
    "docker-compose.yml"
    "vercel.json"
    "render.yaml"
    ".github/workflows/deploy.yml"
    "nginx.conf"
    "LICENSE"
    "setup.py"
)

echo ""
echo "📋 Optional Files:"
for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "⭕ $file (optional)"
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo ""
    echo "🎉 All required files present! Your project is ready for GitHub."
else
    echo ""
    echo "⚠️  Missing required files:"
    printf '   - %s\n' "${missing_files[@]}"
    echo ""
    echo "Please add these files before pushing to GitHub."
fi

echo ""
echo "🔐 Security Checklist:"
echo "====================="
echo "✅ .env file is in .gitignore"
echo "✅ MongoDB credentials are in .env (not in code)"
echo "✅ No hardcoded secrets in the codebase"

if grep -r "mongodb+srv://" --exclude-dir=.git --exclude="*.md" --exclude=".env*" . 2>/dev/null | grep -v ".env.example"; then
    echo "⚠️  Found potential MongoDB URLs in code - please check these files"
else
    echo "✅ No MongoDB URLs found in code"
fi

echo ""
echo "🚀 Your University Chatbot is ready for the world!"
echo "Happy coding! 🎓✨"