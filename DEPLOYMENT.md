# Backend Deployment Guide

## Deploy to Railway (Recommended)

1. **Sign up/Login to Railway**: Visit [railway.app](https://railway.app)

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account and select this repository

3. **Configure**:
   - Railway will automatically detect `railway.toml` and configure the build
   - Wait for deployment to complete (5-10 minutes for first build)

4. **Get Your URL**:
   - Go to Settings → Domains
   - Railway will provide a URL like: `https://your-app.up.railway.app`

5. **Configure Frontend**:
   - In Lovable, go to Project Settings → Environment Variables
   - Add: `VITE_API_URL` = `https://your-app.up.railway.app`
   - Click "Update" in the publish dialog to redeploy frontend

---

## Deploy to Render

1. **Sign up/Login to Render**: Visit [render.com](https://render.com)

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select this repository

3. **Configure**:
   - **Name**: ai-text-detector-backend
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Choose "Free" or "Starter" (Free may have cold starts)

4. **Get Your URL**:
   - After deployment, Render provides a URL like: `https://your-app.onrender.com`

5. **Configure Frontend**:
   - In Lovable, go to Project Settings → Environment Variables
   - Add: `VITE_API_URL` = `https://your-app.onrender.com`
   - Click "Update" in the publish dialog to redeploy frontend

---

## Deploy to Heroku

1. **Install Heroku CLI**: [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login and Create App**:
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

4. **Configure Frontend**:
   - Get your Heroku URL: `https://your-app-name.herokuapp.com`
   - In Lovable, add `VITE_API_URL` environment variable
   - Redeploy frontend

---

## Important Notes

- **First deployment** takes 5-10 minutes (downloading ML models)
- **Cold starts** on free tiers may cause initial request delays
- **Memory**: Ensure at least 1GB RAM (512MB may cause OOM errors)
- **Health check**: All platforms use `/health` endpoint

## Testing Your Deployment

Once deployed, test the backend:

```bash
curl https://your-backend-url.com/health
# Should return: {"status":"healthy"}

curl -X POST https://your-backend-url.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a test sentence."}'
# Should return analysis results
```
