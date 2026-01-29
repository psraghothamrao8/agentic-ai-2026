import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import zipfile
import io
import os
import threading

# Minimal Imports
from app.config import BASE_DIR, MODELS_DIR, get_data_paths
from app.services.agent_service import analyze_situation_and_decide
from app.services.data_service import apply_fix, get_dataset_stats
from app.services.training_service import run_training

app = FastAPI(title="AutoML Agent (n8n)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dataset (Dynamic)
# We wrap this in a middleware or check on request? 
# For StaticFiles, it needs a directory at startup.
# We will point it to the ROOT dataset folder, so we can access /dataset/{dataset_name}/...
paths = get_data_paths() # Initial load
app.mount("/dataset", StaticFiles(directory=paths["dataset_root"]), name="dataset")

# --- State ---
training_state = {"status": "idle", "result": None, "error": None}
state_lock = threading.Lock()

# --- Models ---
class FixRequest(BaseModel):
    file_path: str
    action: str
    new_label: Optional[str] = None

class BatchFixRequest(BaseModel):
    file_paths: list[str]
    action: str
    new_label: Optional[str] = None

# --- Routes ---

@app.get("/api/status")
def get_status():
    """Polled by n8n to check system state."""
    return {"dataset": get_dataset_stats(), "training": training_state}

@app.get("/api/analyze")
def analyze():
    """Step 1 in n8n flow: Analyze dataset."""
    return analyze_situation_and_decide()

@app.post("/api/batch_fix")
def batch_fix(req: BatchFixRequest):
    """Step 2 in n8n flow: Apply AI suggestions."""
    results = [{"path": p, "success": s, "message": m} 
               for p in req.file_paths 
               for s, m in [apply_fix(p, req.action, req.new_label)]]
    success_count = sum(1 for r in results if r["success"])
    return {"status": "completed", "success": success_count, "results": results}

@app.post("/api/train")
def start_train():
    """Step 3 in n8n flow: Train model."""
    if training_state["status"] == "running": raise HTTPException(400, "Already running")
    
    def task():
        with state_lock: training_state.update({"status": "running", "result": None, "error": None})
        try:
            res = run_automated_training()
            with state_lock: training_state.update({"status": "completed", "result": res})
        except Exception as e:
            with state_lock: training_state.update({"status": "failed", "error": str(e)})

    threading.Thread(target=task).start()
    return {"status": "started"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Optional: Upload dataset via n8n."""
    if not file.filename.endswith(".zip"): raise HTTPException(400, "Zip only")
    try:
        if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR)
        content = await file.read()
        with zipfile.ZipFile(io.BytesIO(content)) as z: z.extractall(DATASET_DIR)
        return {"status": "success"}
    except Exception as e: raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
