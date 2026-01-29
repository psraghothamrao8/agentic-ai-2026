import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api import routes
from backend.app.core.config import DATASET_DIR, BASE_DIR
import os

app = FastAPI(title="AutoML Agent")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router
app.include_router(routes.router, prefix="/api")

# Mount Dataset for images
app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

# Static Files (Frontend)
# backend/app/main.py -> app -> backend -> root -> frontend
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
