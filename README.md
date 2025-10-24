# EstateAI-Backend

This is a FastAPI-based backend for advanced real estate price prediction.

This repository is prepared to deploy to Render (recommended) or any container platform.

## Quick deploy to Render (recommended)

1. Sign in to https://render.com and connect your GitHub account.
2. Create a new **Web Service** and select this repository (`arijit3111w/EstateAI-Backend`).
3. Configure the service:
   - Branch: `main`
   - Build command: (leave empty) or `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.x (Render auto-detects)
4. Add environment variables in Render's settings:
   - `GEMINI_API_KEY` — your Google Generative AI (Gemini) API key
   - Any other secrets (do NOT commit `.env` to GitHub)
5. Deploy and watch the logs. After deployment, visit the service URL and open `/` and `/health`.

## Notes and caveats

- Model artifact size: `model/artifacts_v2/xgb_model_advanced.joblib` is included and is ~75 MB — under GitHub's single-file limit (100 MB). If you later add larger files, use Git LFS or external storage.
- Secrets: keep `GEMINI_API_KEY` out of git. Use Render's environment variable UI.
- If the app fails on startup due to missing model files, make sure the `model/artifacts_v2` directory is present on the server. Alternatively, host model files in cloud storage and download them at startup (I can help add that logic).

## Local run (development)

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run locally:

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Files added for deployment
- `Procfile` — start command for Render/Heroku
- `.env.example` — example env variables (do NOT commit `.env`)

If you want, I can also add a `Dockerfile` and a GitHub Actions workflow to auto-deploy to Render or to build/push a container image.
