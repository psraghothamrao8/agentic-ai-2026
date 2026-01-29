import os
import pathlib

# Centralized BASE_DIR resolution
# This should point to the root of the project (agent-ai-2035)
# backend/app/core/config.py -> core -> app -> backend -> agent-ai-2035
BASE_DIR = pathlib.Path(__file__).parent.parent.parent.parent.resolve()

DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

# Dynamic detection of dataset folder name
# We look for a subdirectory in 'dataset/' to use as the active dataset.
# Default to 'mlcc' if nothing enters.
DATASET_NAME = "mlcc"

if os.path.exists(DATASET_ROOT):
    # Get all subdirectories
    subdirs = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
    # Filter out common artifacts
    valid_subdirs = [d for d in subdirs if not d.startswith('.') and d not in ['__pycache__']]
    
    if len(valid_subdirs) > 0:
        # If we find a directory named 'mlcc', prefer it (legacy compatibility)
        if "mlcc" in valid_subdirs:
            DATASET_NAME = "mlcc"
        else:
            # Otherwise use the first valid subdirectory found (e.g. 'my_custom_data')
            DATASET_NAME = valid_subdirs[0]

DATASET_DIR = os.path.join(DATASET_ROOT, DATASET_NAME)
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
