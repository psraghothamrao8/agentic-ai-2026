import os
import glob
import shutil
import numpy as np
from PIL import Image
from app.config import get_data_paths, MODELS_DIR

# --- Minimal Imports ---
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from cleanlab.filter import find_label_issues
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"

if HAS_TORCH:
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    val_transform = None
    train_transform = None

# --- Dataset ---
class CustomImageDataset(Dataset if HAS_TORCH else object):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        if os.path.exists(root_dir):
            self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            for cls in self.classes:
                msg_files = glob.glob(os.path.join(root_dir, cls, "**", "*.*"), recursive=True)
                for f in msg_files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.files.append(f)
                        self.labels.append(self.class_to_idx[cls])
        else:
            self.classes = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.files[idx]).convert("RGB")
            if HAS_TORCH and self.transform: image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224)) if HAS_TORCH else None
        return image, self.labels[idx], self.files[idx]

def get_dataset_stats():
    paths = get_data_paths()
    stats = {}
    for split, key in [("train", "train"), ("val", "val")]:
        path = paths[key]
        if not os.path.exists(path):
             stats[split] = {"count": 0}
             continue
        count = sum([len(files) for r, d, files in os.walk(path)])
        stats[split] = {"count": count}
    return stats

# --- Minimal Detection Logic ---

def extract_features(dataset):
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(DEVICE).eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            feats.append(model(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(feats)

def detect_issues_in_split(split, path):
    if not HAS_CLEANLAB or not HAS_TORCH or not os.path.exists(path): return []
    ds = CustomImageDataset(path, transform=val_transform)
    if len(ds) < 10: return []
    
    feats = extract_features(ds)
    probs = cross_val_predict(LogisticRegression(max_iter=500), feats, np.array(ds.labels), cv=3, method="predict_proba")
    
    try:
        issues_idx = find_label_issues(labels=ds.labels, pred_probs=probs, return_indices_ranked_by="self_confidence")
    except: return []

    return [{
        "file_path": ds.files[i],
        "issue_type": "label_issue",
        "suggested_label": ds.classes[np.argmax(probs[i])],
        "split": split
    } for i in issues_idx]

def detect_issues_with_model(model_path, split, path):
    if not HAS_CLEANLAB or not HAS_TORCH or not os.path.exists(model_path): return []
    ds = CustomImageDataset(path, transform=val_transform)
    if len(ds) < 10: return []

    # 1. Get Model Probabilities (85% weight)
    from app.services.training_service import create_model
    model = create_model(len(ds.classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs_model = []
    with torch.no_grad():
        for imgs, _, _ in DataLoader(ds, batch_size=256, shuffle=False):
            probs_model.append(F.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy())
    probs_model = np.vstack(probs_model)

    # 2. Get LogReg Probabilities (15% weight - Normalized from user request)
    # Using cross-validation on the split itself to check for internal consistency
    feats = extract_features(ds)
    try:
        probs_logreg = cross_val_predict(LogisticRegression(max_iter=500), feats, np.array(ds.labels), cv=3, method="predict_proba")
    except:
        # Fallback if CV fails (e.g. too few samples per class)
        probs_logreg = probs_model 

    # 3. Weighted Ensemble
    # User requested ~85% Model, ~25% LogReg (110%). Normalizing to 0.85 / 0.15.
    probs_final = 0.85 * probs_model + 0.15 * probs_logreg

    try:
        issues_idx = find_label_issues(labels=ds.labels, pred_probs=probs_final, return_indices_ranked_by="self_confidence")
    except: return []

    return [{
        "file_path": ds.files[i],
        "issue_type": "hybrid_label_issue",
        "suggested_label": ds.classes[np.argmax(probs_final[i])],
        "split": split
    } for i in issues_idx]

def detect_issues():
    """
    Hybrid detection:
    - Train: Always uses Feature Extraction + Logistic Regression (Cleanlab default).
    - Val/Test: Prefers 'best_model.pth' if available (better for finding model-specific errors).
    """
    paths = get_data_paths()
    train_dir = paths["train"]
    val_dir = paths["val"]
    test_dir = paths["test"]
    
    issues = []
    
    # 1. Analyze Train (Always LogReg)
    # Rationale: We don't want to bias training data issues with a model trained on them? 
    # Or simply because LogReg is robust for the initial pass.
    if os.path.exists(train_dir):
        issues.extend(detect_issues_in_split("train", train_dir))
        
    # 2. Analyze Val/Test (Model if available, else LogReg)
    model_path = os.path.join(MODELS_DIR, "best_model.pth")
    has_model = os.path.exists(model_path)
    
    for split, path in [("val", val_dir), ("test", test_dir)]:
        if not os.path.exists(path): continue
        
        if has_model:
            # Use specific model predictions
            issues.extend(detect_issues_with_model(model_path, split, path))
        else:
            # Fallback to generic features
            issues.extend(detect_issues_in_split(split, path))
            
    return {"issues": issues}

def apply_fix(path, action, label=None):
    if not os.path.exists(path): return False, "Not found"
    try:
        if action == 'delete':
            os.remove(path)
        elif action == 'move' and label:
            dest = os.path.join(os.path.dirname(os.path.dirname(path)), label, os.path.basename(path))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(path, dest)
        return True, "Done"
    except Exception as e: return False, str(e)
