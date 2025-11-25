# Deploying to Render

This guide explains how to deploy the backend API to [Render](https://render.com).

## Prerequisites

1. A [Render account](https://dashboard.render.com/register)
2. Your code pushed to GitHub/GitLab/Bitbucket
3. At least 512MB RAM instance (the model requires memory optimizations)

## Quick Deploy

### Option 1: Using Render Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com) and click **New > Web Service**
2. Connect your GitHub/GitLab/Bitbucket repository
3. Configure the service:

   **Basic Settings:**
   - **Name**: `ai-text-detector-backend` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Language**: `Python 3`
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && python main.py`

   **Instance Type:**
   - **Free**: 512MB RAM (may work with optimizations)
   - **Starter**: 512MB RAM ($7/month) - Recommended
   - **Standard**: 1GB RAM ($25/month) - More reliable

   **Environment Variables:**
   - No additional variables needed (PORT is set automatically by Render)

4. Click **Create Web Service**

### Option 2: Using render.yaml (Infrastructure as Code)

The project includes a `render.yaml` file for automated deployment. After connecting your repo, Render will automatically detect and use this configuration.

## Port Binding

According to [Render's documentation](https://render.com/docs/web-services#port-binding):

- ✅ The backend **already binds to `0.0.0.0`** (required)
- ✅ Uses `PORT` environment variable (Render sets this automatically, default is `10000`)
- ✅ Falls back to `5000` for local development

## Health Check

The service includes a `/health` endpoint for Render's health checks:

```bash
GET https://your-service.onrender.com/health
```

Response:
```json
{"status": "healthy"}
```

## Memory Optimizations

The backend is optimized for <512MB hosting:

- **Float16 precision**: Reduces model memory by ~50%
- **Low CPU memory usage**: Reduces peak memory during loading
- **Gradient cleanup**: Clears memory after each inference
- **Garbage collection**: Automatic cleanup after requests

## Troubleshooting

### Out of Memory Errors

If you see "Out of memory" errors:

1. **Upgrade instance**: Use Starter (512MB) or Standard (1GB) instead of Free
2. **Check logs**: Go to your service's **Logs** tab in Render Dashboard
3. **Reduce max_length**: The API accepts `max_length` parameter (default 512)

### Model Loading Fails

If model loading fails:

1. **Check build logs**: Ensure all dependencies install correctly
2. **Verify safetensors**: The model should load with safetensors format
3. **Check Hugging Face access**: Ensure the model `andreas122001/roberta-mixed-detector` is accessible

### Service Not Responding

1. **Check health endpoint**: `GET /health` should return `{"status": "healthy"}`
2. **Verify port binding**: Ensure the service binds to `0.0.0.0` (already configured)
3. **Check logs**: Look for startup errors in Render Dashboard

## Environment Variables

Render automatically sets:
- `PORT`: The port your service should bind to (default: `10000`)
- `RENDER`: Set to indicate running on Render (disables debug mode)

Optional variables you can set:
- `HF_TOKEN`: Hugging Face token for private models (not needed for public models)

## Connecting Frontend

After deployment, update your frontend to use the Render URL:

```typescript
// In your frontend config
const API_URL = import.meta.env.VITE_API_URL || "https://your-service.onrender.com";
```

Or set `VITE_API_URL` environment variable in your frontend deployment.

## Cost Estimation

- **Free Tier**: 512MB RAM, may work with optimizations (limited hours/month)
- **Starter**: 512MB RAM, $7/month - Recommended for production
- **Standard**: 1GB RAM, $25/month - More reliable for high traffic

## References

- [Render Web Services Documentation](https://render.com/docs/web-services)
- [Render Port Binding](https://render.com/docs/web-services#port-binding)
- [Render Python Support](https://render.com/docs/python)

