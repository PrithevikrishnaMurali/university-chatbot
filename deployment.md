# 🚀 Deployment Guide

This guide will help you deploy your University Recommendation Chatbot to various platforms.

## 📋 Pre-deployment Checklist

- [ ] MongoDB Atlas cluster is set up and running
- [ ] Environment variables are configured
- [ ] Code is tested locally
- [ ] All sensitive data is in `.env` file (not committed to Git)

## 🎯 Quick Deployment Options

### Option 1: Heroku (Easiest) ⭐️

**Perfect for: Beginners, quick demos**

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Windows
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-chatbot-name
   ```

3. **Set Environment Variables**
   ```bash
   heroku config:set MONGODB_URL="your_mongodb_url_here"
   heroku config:set DATABASE_NAME="University_data"
   heroku config:set DEBUG="False"
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open Your App**
   ```bash
   heroku open
   ```

**Pros**: Easy setup, automatic HTTPS, good for demos
**Cons**: Free tier has limitations, can sleep after 30 minutes

---

### Option 2: Railway (Fast & Modern) 🚄

**Perfect for: Modern deployment, good performance**

1. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**
   ```bash
   railway login
   railway init
   railway up
   ```

3. **Set Environment Variables** (via Railway dashboard)
   - Go to your Railway project
   - Add: `MONGODB_URL`, `DATABASE_NAME`, `DEBUG=False`

**Pros**: Modern interface, good performance, easy scaling
**Cons**: Requires credit card for continued use

---

### Option 3: Render (Free Tier Available) 🎨

**Perfect for: Free hosting, reliable**

1. **Connect GitHub Repository**
   - Go to [render.com](https://render.com)
   - Connect your GitHub account
   - Select your repository

2. **Configure Web Service**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`

3. **Set Environment Variables**
   ```
   MONGODB_URL = your_mongodb_url_here
   DATABASE_NAME = University_data
   DEBUG = False
   PORT = 8000
   ```

**Pros**: Free tier available, reliable, automatic deployments
**Cons**: Free tier has limitations

---

### Option 4: Vercel (Serverless) ▲

**Perfect for: Serverless deployment, global CDN**

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Deploy**
   ```bash
   vercel login
   vercel --prod
   ```

3. **Set Environment Variables** (via Vercel dashboard)
   - Add: `MONGODB_URL`, `DATABASE_NAME`, `DEBUG=False`

**Pros**: Global CDN, serverless, excellent performance
**Cons**: Cold starts, function timeout limits

---

### Option 5: Docker + Any VPS 🐳

**Perfect for: Full control, custom servers**

1. **Build Docker Image**
   ```bash
   docker build -t university-chatbot .
   ```

2. **Run Locally** (test first)
   ```bash
   docker run -p 8000:8000 --env-file .env university-chatbot
   ```

3. **Deploy to VPS**
   ```bash
   # Push to Docker Hub
   docker tag university-chatbot yourusername/university-chatbot
   docker push yourusername/university-chatbot
   
   #