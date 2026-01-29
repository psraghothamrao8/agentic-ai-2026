from fastapi import APIRouter, HTTPException, Response, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import csv
import shutil
import zipfile
from backend.app.services.agent import analyze_situation_and_decide, diagnose_after_exploration
from backend.app.services.data import apply_fix, get_dataset_stats, CustomImageDataset, detect_issues_with_model
from backend.app.core.config import TRAIN_DIR, MODELS_DIR, DATASET_DIR, VAL_DIR, TEST_DIR
from backend.app.services.training import run_automated_training, auto_explore
from backend.app.schemas.requests import FixRequest, BatchFixRequest, BatchSuggestionRequest
import threading
import os
import json

router = APIRouter()

# Simple In-Memory State for Training
training_state = {
    "status": "idle",
    "progress": [],
    "result": None
}

# Global Lock for Thread Safety
state_lock = threading.Lock()

# Phase 4: Auto-Training State
auto_training_state = {
    "status": "idle",
    "current_config": 0,
    "total_configs": 0,
    "current_trial": 0,
    "total_trials": 0,
    "best_acc": 0.0,
    "exploration_results": None,
    "diagnosis": None,
    "iteration": 0,
    "max_iterations": 3,
    "current_epoch": 0,
    "total_epochs": 0
}

def restore_state():
    global auto_training_state
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    
    # Check if dataset exists. If not, we SHOULD NOT restore state.
    dataset_exists = os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0
    
    if os.path.exists(metrics_path) and dataset_exists:
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                # Load all results for the leaderboard
                if "all_results" in data:
                    auto_training_state["results"] = data["all_results"]
                
                # If we have a valid summary, mark as completed or partial
                if "status" in data:
                    if data["status"] == "success":
                        auto_training_state["status"] = "completed"
                    elif data["status"] == "partial":
                        auto_training_state["status"] = "exploring" # Or keep exploring if it was in progress
                    
                    auto_training_state["exploration_results"] = {
                        "status": data["status"],
                        "best_result": data.get("best_result"),
                        "all_results": data.get("all_results", [])
                    }
                    if data.get("best_result"):
                        auto_training_state["best_acc"] = data["best_result"].get("val_acc", 0.0)
        except Exception as e:
            print(f"Failed to restore state: {e}")

restore_state()

@router.get("/status")
@router.get("/status")
def get_system_status():
    with state_lock:
        return {
            "dataset_stats": get_dataset_stats(),
            "training_state": training_state,
            "auto_training_state": auto_training_state
        }

@router.get("/debug_config")
def debug_server_config():
    """Returns the server's current path configuration for debugging."""
    return {
        "BASE_DIR": str(BASE_DIR),
        "DATASET_DIR": str(DATASET_DIR),
        "TRAIN_DIR": str(TRAIN_DIR),
        "Dataset_Exists": os.path.exists(DATASET_DIR),
        "Train_Exists": os.path.exists(TRAIN_DIR),
        "Dataset_Content": os.listdir(DATASET_DIR) if os.path.exists(DATASET_DIR) else "N/A"
    }

@router.get("/analyze")
def analyze_dataset():
    """
    Triggers the Agent to analyze the dataset.
    """
    decision = analyze_situation_and_decide()
    return decision

@router.get("/analyze_with_model")
def analyze_dataset_with_model():
    """
    Triggers hybrid analysis using the best trained model.
    """
    # 1. Identify best model path
    model_path = None
    if auto_training_state.get("best_acc", 0) > 0:
        # Check exploration results for path
        if auto_training_state.get("exploration_results") and auto_training_state["exploration_results"].get("best_result"):
             model_path = auto_training_state["exploration_results"]["best_result"].get("model_path")
    
    # Fallback to default best_model.pth if state is lost but file exists
    if not model_path:
        default_path = os.path.join(MODELS_DIR, "best_model.pth")
        if os.path.exists(default_path):
            model_path = default_path
            
    if not model_path or not os.path.exists(model_path):
         raise HTTPException(status_code=400, detail="No trained model found. Please run benchmarking first.")
         
    # 2. Run Analysis
    issues = []
    # Train
    issues.extend(detect_issues_with_model(model_path, "train", TRAIN_DIR))
    # Val
    issues.extend(detect_issues_with_model(model_path, "val", VAL_DIR))
    # Test
    if os.path.exists(TEST_DIR):
        issues.extend(detect_issues_with_model(model_path, "test", TEST_DIR))
    
    if isinstance(issues, dict) and "error" in issues:
         # Backward compat if function returns error dict (it currently returns list or empty list)
         pass 
         
    # 3. Construct decision object similar to regular analysis
    return {
        "analysis": f"Hybrid Analysis using model: {os.path.basename(model_path)}. Precision is higher than initial analysis.",
        "recommended_action": "data_cleaning",
        "issues_list": issues,
        "raw_issues_count": len(issues),
        "decision": "HYBRID_ANALYSIS"
    }

@router.get("/evaluate_current_model")
def evaluate_current_model_endpoint():
    """
    Evaluates the current best model and returns metrics including Miss and Overkill rates.
    """
    from backend.app.services.training import evaluate_saved_model
    results = evaluate_saved_model()
    if "error" in results:
        raise HTTPException(status_code=400, detail=results["error"])
    return results

@router.get("/get_classes")
def get_available_classes():
    """
    Returns all available classes for the dropdown.
    """
    if not os.path.exists(TRAIN_DIR):
        return {"classes": []}
    
    dataset = CustomImageDataset(TRAIN_DIR)
    return {"classes": dataset.classes}

@router.post("/fix_issue")
def fix_data_issue(req: FixRequest):
    success, msg = apply_fix(req.file_path, req.action, req.new_label)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"status": "success", "message": msg}

@router.post("/batch_fix_suggestions")
def batch_fix_suggestions(req: BatchSuggestionRequest):
    """
    Apply varying fixes (move or delete) for multiple files.
    """
    results = []
    
    for item in req.items:
        # Action is 'delete' if suggestion is 'delete', else 'move'
        action = 'delete' if item.new_label.lower() == 'delete' else 'move'
        success, msg = apply_fix(item.file_path, action, item.new_label)
        results.append({"path": item.file_path, "success": success, "message": msg})
        
    success_count = sum(1 for r in results if r["success"])
    return {
        "status": "completed",
        "total": len(req.items),
        "success": success_count,
        "results": results
    }

@router.post("/batch_fix")
def batch_fix_issues(req: BatchFixRequest):
    """
    Apply the same fix to multiple files.
    """
    results = []
    for path in req.file_paths:
        success, msg = apply_fix(path, req.action, req.new_label)
        results.append({"path": path, "success": success, "message": msg})
    
    success_count = sum(1 for r in results if r["success"])
    return {
        "status": "completed",
        "total": len(req.file_paths),
        "success": success_count,
        "failed": len(req.file_paths) - success_count,
        "results": results
    }

@router.post("/download_issues_csv")
def download_issues_csv(req: BatchFixRequest):
    """
    Generates a CSV file of the provided issues.
    Reuses BatchFixRequest just to get the list of file_paths, 
    but ideally we want the full issue details. 
    Let's use a generic dict body or rely on the frontend sending the right structure.
    For simplicity, let's accept a list of issue objects.
    """
    # Since we don't have a specific schema for "Issue" in requests.py yet,
    # and BatchFixRequest only has file_paths, let's define a quick Pydantic model here or just accept dict.
    pass

from pydantic import BaseModel
class IssueItem(BaseModel):
    file_path: str
    issue_type: str
    severity: str = "MEDIUM"  # Default for backward compatibility
    quality_score: float = 0.5  # Default for backward compatibility
    given_label: str
    suggested_label: str
    confidence: float
    split: str
    details: dict = {}  # Optional details field

class CsvDownloadRequest(BaseModel):
    issues: list[IssueItem]

@router.post("/download_issues_csv_file")
def download_issues_csv_file(req: CsvDownloadRequest):
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Enhanced header with new fields
    writer.writerow(["File Name", "Split", "Severity", "Quality Score", "Actual Label", "Suggested Label", "Confidence", "Issue Type", "Full Path"])
    
    for issue in req.issues:
        writer.writerow([
            os.path.basename(issue.file_path),
            issue.split,
            getattr(issue, 'severity', 'MEDIUM'),
            f"{getattr(issue, 'quality_score', 0.5):.3f}",
            issue.given_label,
            issue.suggested_label,
            f"{issue.confidence:.4f}",
            issue.issue_type,
            issue.file_path
        ])
        
    output.seek(0)
    
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=detected_issues.csv"
    return response

def run_training_background():
    global training_state
def run_training_background():
    global training_state
    with state_lock:
        training_state["status"] = "running"
        training_state["progress"] = []
    
    try:
        result = run_automated_training()
        with state_lock:
            training_state["result"] = result
            training_state["status"] = "completed"
    except Exception as e:
        with state_lock:
            training_state["status"] = "failed"
            training_state["error"] = str(e)

@router.post("/start_training")
def start_training_endpoint():
    if training_state["status"] == "running":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    t = threading.Thread(target=run_training_background)
    t.start()
    
    return {"status": "started", "message": "Training started in background"}

@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a zip file containing the dataset, clears the existing dataset,
    and extracts the new one.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    # Clear existing dataset directory
    try:
        if os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        os.makedirs(DATASET_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear old dataset: {str(e)}")

    # Save and extract zip file
    try:
        parent_dir = os.path.dirname(DATASET_DIR) # .../dataset/
        os.makedirs(parent_dir, exist_ok=True)
        
        # Creates a temporary directory for extraction
        temp_extract_dir = os.path.join(parent_dir, "temp_extract_svc")
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        os.makedirs(temp_extract_dir)

        zip_path = os.path.join(parent_dir, "temp_upload.zip")
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract fully to temp dir
        # Extract fully to temp dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Security: Zip Slip Protection
            for member in zip_ref.namelist():
                # Resolve the absolute path of the extraction
                abs_target = os.path.abspath(os.path.join(temp_extract_dir, member))
                abs_root = os.path.abspath(temp_extract_dir)
                # Check if the extraction path is within the target directory
                if not abs_target.startswith(abs_root):
                    raise HTTPException(status_code=400, detail="Security Error: Zip Slip vulnerability detected in archive.")
            
            zip_ref.extractall(temp_extract_dir)
        
        # Smart Search: Find the directory containing 'train' (case-insensitive)
        actual_data_root = None
        for root, dirs, files in os.walk(temp_extract_dir):
            if any(d.lower() in ['train', 'training', 'train_data', 'train_images'] for d in dirs):
                actual_data_root = root
                break
        
        if actual_data_root:
            print(f"DEBUG: Found dataset root at {actual_data_root}")
            
            # Normalize and Move content to DATASET_DIR
            if not os.path.exists(DATASET_DIR):
                os.makedirs(DATASET_DIR)
                
            for item in os.listdir(actual_data_root):
                src_path = os.path.join(actual_data_root, item)
                
                # Determine destination name (Standardize to lowercase 'train', 'val', 'test')
                item_clean = item.strip()
                item_lower = item_clean.lower()
                destination_name = item_clean # Default to cleaned original name
                
                if item_lower in ['train', 'training', 'train_data', 'train_images']:
                    destination_name = 'train'
                elif item_lower in ['val', 'validation', 'valid', 'val_data', 'val_images']:
                    destination_name = 'val'
                elif item_lower in ['test', 'testing', 'tests', 'test_data', 'test_images']:
                    destination_name = 'test'
                    
                dst_path = os.path.join(DATASET_DIR, destination_name)
                
                # If destination exists (e.g. __MACOSX artifacts), skip or merge
                if os.path.exists(dst_path):
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
        else:
            # Cleanup and fail
            shutil.rmtree(temp_extract_dir)
            os.remove(zip_path)
            return {
                "status": "error", 
                "message": "Structure invalid: Could not find a 'train' folder (case-insensitive) anywhere in the zip file."
            }
        
        # Cleanup temp resources
        shutil.rmtree(temp_extract_dir)
        os.remove(zip_path)
        
        # Final Verification
        if not os.path.exists(TRAIN_DIR):
             return {
                "status": "warning", 
                "message": "Extraction finished but 'train' folder is missing from destination."
            }
        
        return {"status": "success", "message": "Dataset uploaded and verified successfully"}
    except Exception as e:
        # Emergency cleanup
        if 'temp_extract_dir' in locals() and os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        if 'zip_path' in locals() and os.path.exists(zip_path):
            os.remove(zip_path)
        raise HTTPException(status_code=500, detail=f"Failed to process zip file: {str(e)}")

# ============== Phase 4: Auto-Training Endpoints ==============

def run_auto_exploration_background():
    """Background thread for auto-exploration."""
    global auto_training_state
    
    try:
        with state_lock:
            auto_training_state["status"] = "exploring"
            auto_training_state["iteration"] += 1
        
        print(f"\n[AUTO] Starting auto-exploration (Iteration {auto_training_state['iteration']})...")
        
        def progress_cb(status_dict):
            global auto_training_state
            with state_lock:
                auto_training_state.update(status_dict)
            
        # Run exploration with callback
        results = auto_explore(target_accuracy=0.90, max_time_hours=2, progress_callback=progress_cb)
        
        with state_lock:
            auto_training_state["exploration_results"] = results
        
        if results["status"] == "failed":
            error_msg = results.get('error', 'Unknown trainer error')
            print(f"[ERROR] Trainer returned failure: {error_msg}")
            raise Exception(error_msg)

        if results["status"] == "success":
            # Success! Training achieved target
            with state_lock:
                auto_training_state["status"] = "completed"
                if results.get("best_result") and "val_acc" in results["best_result"]:
                    auto_training_state["best_acc"] = results["best_result"]["val_acc"]
        else:
            # Need diagnosis
            with state_lock:
                auto_training_state["status"] = "diagnosing"
            
            diagnosis = diagnose_after_exploration(results)
            
            with state_lock:
                auto_training_state["diagnosis"] = diagnosis
                
                # After diagnosis, transition to completed state
                # (Diagnosis is informational only, training is done)
                auto_training_state["status"] = "completed"
                if results.get("best_result") and "val_acc" in results["best_result"]:
                    auto_training_state["best_acc"] = results["best_result"]["val_acc"]
            
            # Legacy logic for special cases (kept for reference but won't execute now)
            # Check if we should ask user or continue
            # if diagnosis["diagnosis"] in ["data_quality"]:
            #     # Ask user to clean data
            #     auto_training_state["status"] = "waiting_user"
            # elif auto_training_state["iteration"] >= auto_training_state["max_iterations"]:
            #     # Max iterations reached
            #     auto_training_state["status"] = "completed"
            #     auto_training_state["best_acc"] = results["best_result"]["val_acc"]
            # else:
            #     # Continue with another iteration (shouldn't happen often)
            #     auto_training_state["status"] = "completed"
                
    except Exception as e:
        with state_lock:
            auto_training_state["status"] = "failed"
            auto_training_state["error"] = str(e)
        print(f"[ERROR] Auto-exploration failed: {e}")
        import traceback
        traceback.print_exc()

@router.post("/start_auto_training")
def start_auto_training():
    """Starts the smart auto-exploration process."""
    global auto_training_state
    
    if auto_training_state["status"] in ["exploring", "diagnosing"]:
        current_status = auto_training_state["status"]
        raise HTTPException(
            status_code=400, 
            detail=f"Auto-training already in progress (status: {current_status}). Please wait for completion or refresh."
        )
    
    
    # Reset state by clearing and updating (maintain reference)
    with state_lock:
        auto_training_state.clear()
        auto_training_state.update({
            "status": "exploring",
            "current_config": 0,
            "total_configs": 0,
            "current_trial": 0,
            "total_trials": 0,
            "results": [],
            "best_acc": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "exploration_results": None,
            "diagnosis": None,
            "iteration": 0,
            "max_iterations": 3
        })
    
    t = threading.Thread(target=run_auto_exploration_background)
    t.start()
    
    return {"status": "started", "message": "Auto-training started"}

@router.get("/auto_training_status")
def get_auto_training_status():
    """Returns current auto-training state for frontend polling."""
    # Include stats for dashboard sync
    with state_lock:
        state = auto_training_state.copy()
        
    state["dataset_stats"] = get_dataset_stats()
    return state

@router.post("/user_feedback")
def handle_user_feedback(action: str):
    """
    Handles user feedback after diagnosis.
    action: 'recleaned' | 'satisfied' | 'continue'
    """
    global auto_training_state
    
    # Validate action parameter
    valid_actions = ["recleaned", "satisfied", "continue"]
    if action not in valid_actions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
        )
    
    if action == "recleaned":
        # User cleaned data, restart exploration
        with state_lock:
            auto_training_state["status"] = "exploring"
        t = threading.Thread(target=run_auto_exploration_background)
        t.start()
        return {"status": "restarted", "message": "Restarting exploration with cleaned data"}
    elif action == "satisfied":
        # User is satisfied, mark as complete
        with state_lock:
            auto_training_state["status"] = "completed"
        return {"status": "completed", "message": "Marked as complete"}
    else:
        return {"status": "unknown_action"}

@router.post("/reset_training_state")
def reset_training_state(force_delete: bool = False, hard_reset: bool = False):
    """
    Resets the auto-training state.
    - Soft Reset (default): Resets state to idle.
    - Hard Reset (hard_reset=True): Resets state AND deletes all Datasets, Models, and Logs.
    """
    global auto_training_state
    
    current_status = auto_training_state["status"]
    
    # Reset to initial idle state
    with state_lock:
        auto_training_state.clear()
        auto_training_state.update({
            "status": "idle",
            "current_config": 0,
            "total_configs": 0,
            "current_trial": 0,
            "total_trials": 0,
            "best_acc": 0.0,
            "exploration_results": None,
            "diagnosis": None,
            "iteration": 0,
            "max_iterations": 3,
            "current_epoch": 0,
            "total_epochs": 0
        })
    
    deleted_items = []

    # Hard Reset: Delete Everything
    if hard_reset:
        try:
            # 1. Delete Datasets
            if os.path.exists(DATASET_DIR):
                shutil.rmtree(DATASET_DIR)
                os.makedirs(DATASET_DIR)
                deleted_items.append("Datasets")
            
            # 2. Delete Models (Preserve cleanlab cache if strictly needed, but hard means hard)
            if os.path.exists(MODELS_DIR):
                shutil.rmtree(MODELS_DIR)
                os.makedirs(MODELS_DIR)
                deleted_items.append("Models")

            # 3. Delete Logs
            if os.path.exists(LOGS_DIR):
                shutil.rmtree(LOGS_DIR)
                os.makedirs(LOGS_DIR)
                deleted_items.append("Logs")
                
        except Exception as e:
            print(f"Error during hard reset: {e}")
            return {
                 "status": "error",
                 "message": f"Partial reset. Failed to delete some files: {str(e)}"
            }

    # Soft Reset / Legacy force_delete: Just metrics.json
    elif force_delete:
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            try:
                os.remove(metrics_path)
                deleted_items.append("Metrics")
            except Exception as e:
                print(f"Failed to delete metrics.json: {e}")
    
    return {
        "status": "reset", 
        "message": f"System Reset ({'Hard' if hard_reset else 'Soft'}). Deleted: {', '.join(deleted_items) if deleted_items else 'State Only'}",
        "previous_status": current_status
    }
