
import os
import shutil
import glob
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import entropy
try:
    from cleanlab.classification import CleanLearning
    from cleanlab.outlier import OutOfDistribution
    from cleanlab.filter import find_label_issues
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from backend.app.ml.networks import create_model

from backend.app.core.config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, LOGS_DIR, MODELS_DIR

if HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = "cpu"

import numpy as np
from PIL import Image

# Configuration for issue detection
ISSUE_DETECTION_CONFIG = {
    "outlier_percentile": 5,  # Top 5% most outlier-like
    "duplicate_threshold": 0.98,
    "min_confidence_for_relabel": 0.95, # Threshold for suppressing weak suggestions
    "use_ensemble": True,
    "ensemble_weights": {"model": 0.85, "aux": 0.15}, # New: 85% Best Model, 15% Aux
    "severity_thresholds": {
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.5,
        "low": 0.3
    }
}

# Image Transformations
if HAS_TORCH:
    # Standard validation/test transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Augmented training transform
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Added for industrial parts
        transforms.RandomRotation(30),   # Increased from 15 for better orientation robustness
        transforms.RandomGrayscale(p=0.1), # Handle lighting/color variations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
else:
    val_transform = None
    train_transform = None

# Backward compatibility or default
transform = val_transform

class CustomImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        if not os.path.exists(root_dir):
            self.classes = []
            self.class_to_idx = {}
            return
            
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            # Use recursive glob to find images even in subfolders
            # recursive=True requires ** in pattern
            search_pattern = os.path.join(cls_dir, "**", "*.*")
            for img_path in glob.glob(search_pattern, recursive=True): 
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.files.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if HAS_TORCH:
                if self.transform:
                    image = self.transform(image)
                elif transform: # Global fallback
                    image = transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if HAS_TORCH:
                image = torch.zeros((3, 224, 224))
            else:
                # If torch is not available, we can't return a tensor.
                # Returning None might still be unsafe if not handled, but 
                # without torch we are likely not in a training loop.
                image = None
        
        label = self.labels[idx]
        return image, label, img_path

def get_dataset_stats():
    stats = {}
    for split, path in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not os.path.exists(path):
             stats[split] = {"count": 0, "classes": {}}
             continue
        
        classes = {}
        total = 0
        for cls_name in os.listdir(path):
            cls_path = os.path.join(path, cls_name)
            if os.path.isdir(cls_path):
                # Recursive count
                count = 0
                for root, _, files in os.walk(cls_path):
                    count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                classes[cls_name] = count
                total += count
        stats[split] = {"count": total, "classes": classes}
    return stats

def extract_features(dataset: CustomImageDataset):
    """Extracts features using a pre-trained ResNet18."""
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity() # Remove classification layer
    model.eval()
    model = model.to(DEVICE)  # Move model to same device as images
    
    loader = DataLoader(
        dataset, 
        batch_size=256, # Increased for RTX 3090
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    features = []
    
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(DEVICE) if HAS_TORCH else images
            output = model(images)
            features.append(output.cpu().numpy())
            
    return np.vstack(features)

def calculate_prediction_entropy(pred_probs_row):
    """Calculate entropy for a single prediction (higher = more uncertain)."""
    return entropy(pred_probs_row + 1e-10)  # Add small value to avoid log(0)

def calculate_quality_score(pred_probs, labels, features, idx, class_centroids=None):
    """
    Calculate comprehensive quality score for a sample.
    Returns value 0-1 where higher = worse quality (more likely an issue).
    """
    pred_prob_row = pred_probs[idx]
    true_label = labels[idx]
    pred_label = np.argmax(pred_prob_row)
    
    # 1. Confidence score (lower confidence = higher quality score)
    max_confidence = np.max(pred_prob_row)
    confidence_score = 1.0 - max_confidence
    
    # 2. Entropy score (normalized)
    pred_entropy = calculate_prediction_entropy(pred_prob_row)
    max_entropy = np.log(len(pred_prob_row))
    entropy_score = pred_entropy / max_entropy if max_entropy > 0 else 0
    
    # 3. Label mismatch score
    mismatch_score = 1.0 if pred_label != true_label else 0.0
    
    # 4. Feature space isolation (if centroids provided)
    isolation_score = 0.0
    if class_centroids is not None and true_label < len(class_centroids):
        sample_feature = features[idx]
        centroid = class_centroids[true_label]
        # Calculate distance to own class centroid
        distance = np.linalg.norm(sample_feature - centroid)
        # Normalize by average distance within class (simplified - using global std)
        isolation_score = min(distance / (np.std(features) + 1e-6), 1.0)
    
    # Weighted combination
    quality_score = (
        0.4 * confidence_score +
        0.2 * entropy_score +
        0.3 * mismatch_score +
        0.1 * isolation_score
    )
    
    return float(np.clip(quality_score, 0, 1))

def classify_severity(quality_score):
    """Classify issue severity based on quality score."""
    thresholds = ISSUE_DETECTION_CONFIG["severity_thresholds"]
    if quality_score >= thresholds["critical"]:
        return "CRITICAL"
    elif quality_score >= thresholds["high"]:
        return "HIGH"
    elif quality_score >= thresholds["medium"]:
        return "MEDIUM"
    elif quality_score >= thresholds["low"]:
        return "LOW"
    else:
        return "NEGLIGIBLE"

def detect_outliers_ensemble(features, labels=None):
    """
    Detect outliers using ensemble of multiple methods.
    Returns indices of samples identified by at least 2 methods.
    """
    n_samples = len(features)
    if n_samples < 10:
        return np.array([])
    
    outlier_votes = np.zeros(n_samples)
    
    # Method 1: Cleanlab OutOfDistribution
    try:
        ood = OutOfDistribution()
        ood_scores = ood.fit_score(features=features)
        percentile_threshold = ISSUE_DETECTION_CONFIG["outlier_percentile"]
        ood_threshold = np.percentile(ood_scores, percentile_threshold)
        outlier_votes[ood_scores < ood_threshold] += 1
    except Exception as e:
        print(f"OOD detection failed: {e}")
    
    # Method 2: IsolationForest
    try:
        iso_forest = IsolationForest(contamination=percentile_threshold/100, random_state=42)
        iso_predictions = iso_forest.fit_predict(features)
        outlier_votes[iso_predictions == -1] += 1
    except Exception as e:
        print(f"IsolationForest failed: {e}")
    
    # Method 3: LocalOutlierFactor
    try:
        n_neighbors = min(20, n_samples - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=percentile_threshold/100)
        lof_predictions = lof.fit_predict(features)
        outlier_votes[lof_predictions == -1] += 1
    except Exception as e:
        print(f"LOF failed: {e}")
    
    # Return indices with at least 2 votes
    outlier_indices = np.where(outlier_votes >= 2)[0]
    return outlier_indices

def calculate_class_centroids(features, labels, num_classes):
    """Calculate centroid for each class in feature space."""
    centroids = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if np.any(class_mask):
            class_features = features[class_mask]
            centroid = np.mean(class_features, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(features.shape[1]))
    return np.array(centroids)

def get_feature_space_info(features, labels, idx, num_classes, top_k=3):
    """Get contextual information about a sample in feature space."""
    sample_feature = features[idx]
    sample_label = labels[idx]
    
    # Calculate similarity to all other samples
    similarities = cosine_similarity([sample_feature], features)[0]
    similarities[idx] = -1  # Exclude self
    
    # Find most similar samples from the CORRECT class
    correct_class_mask = labels == sample_label
    correct_class_mask[idx] = False  # Exclude self
    
    similar_correct_indices = []
    if np.any(correct_class_mask):
        similarities_correct = similarities.copy()
        similarities_correct[~correct_class_mask] = -1
        similar_correct_indices = np.argsort(similarities_correct)[-top_k:][::-1]
    
    # Calculate distance to class centroid
    centroids = calculate_class_centroids(features, labels, num_classes)
    centroid_distance = np.linalg.norm(sample_feature - centroids[sample_label])
    
    # Normalized distance (relative to average intra-class distance)
    class_mask = labels == sample_label
    if np.sum(class_mask) > 1:
        class_features = features[class_mask]
        avg_distance = np.mean([np.linalg.norm(f - centroids[sample_label]) for f in class_features])
        normalized_distance = centroid_distance / (avg_distance + 1e-6)
    else:
        normalized_distance = 1.0
    
    return {
        "feature_distance": float(centroid_distance),
        "normalized_distance": float(normalized_distance),
        "similar_samples_indices": similar_correct_indices.tolist() if len(similar_correct_indices) > 0 else []
    }

def get_class_issue_summary(all_issues, class_names):
    """Analyze issues per class and return summary statistics."""
    class_stats = {cls: {
        "total_issues": 0,
        "label_issues": 0,
        "outliers": 0,
        "duplicates": 0,
        "avg_quality_score": 0.0,
        "critical_count": 0,
        "confusion_pairs": {}  # Which classes this class is confused with
    } for cls in class_names}
    
    for issue in all_issues:
        given_label = issue["given_label"]
        suggested_label = issue.get("suggested_label", "")
        issue_type = issue["issue_type"]
        quality_score = issue.get("quality_score", 0.0)
        severity = issue.get("severity", "LOW")
        
        if given_label in class_stats:
            class_stats[given_label]["total_issues"] += 1
            
            if "label" in issue_type:
                class_stats[given_label]["label_issues"] += 1
                # Track confusion
                if suggested_label and suggested_label != "delete":
                    if suggested_label not in class_stats[given_label]["confusion_pairs"]:
                        class_stats[given_label]["confusion_pairs"][suggested_label] = 0
                    class_stats[given_label]["confusion_pairs"][suggested_label] += 1
            elif issue_type == "outlier":
                class_stats[given_label]["outliers"] += 1
            elif issue_type == "duplicate":
                class_stats[given_label]["duplicates"] += 1
            
            if severity == "CRITICAL":
                class_stats[given_label]["critical_count"] += 1
    
    # Calculate averages
    for cls in class_names:
        cls_issues = [i for i in all_issues if i["given_label"] == cls]
        if cls_issues:
            avg_score = np.mean([i.get("quality_score", 0.0) for i in cls_issues])
            class_stats[cls]["avg_quality_score"] = float(avg_score)
    
    return class_stats

def detect_issues_in_split(split_name, split_dir):
    """Helper to detect issues in a specific split with enhanced quality scoring."""
    if not HAS_CLEANLAB or not HAS_TORCH:
        print("Cleanlab or Torch not available for issue detection.")
        return []
        
    if not os.path.exists(split_dir):
        return []

    try:
        dataset = CustomImageDataset(split_dir, transform=val_transform)
        if len(dataset) == 0:
            return []
    except Exception as e:
        print(f"Error loading {split_name} dataset for issues: {e}")
        return []

    print(f"Extracting features for {split_name}...")
    features = extract_features(dataset)
    labels = np.array(dataset.labels)
    num_classes = len(dataset.classes)

    # Train a quick LogReg on features to get probs for Cleanlab
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    
    if len(dataset) < 10:
        print(f"Dataset {split_name} too small for cross-val issues detection.")
        return []

    # Calculate class centroids for quality scoring
    class_centroids = calculate_class_centroids(features, labels, num_classes)

    clf = LogisticRegression(max_iter=1000)
    try:
        pred_probs = cross_val_predict(clf, features, labels, cv=min(3, len(dataset)//2), method="predict_proba")
    except Exception as e:
        print(f"Error in cross_val_predict for {split_name}: {e}")
        return []
    
    print(f"Finding label issues in {split_name}...")
    issues_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )
    
    # 2. Outlier Detection (Ensemble)
    print(f"Finding outliers in {split_name} (ensemble method)...")
    outlier_indices = detect_outliers_ensemble(features, labels)
    
    # 3. Near-Duplicate Detection
    print(f"Finding duplicates in {split_name}...")
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, 0)
    duplicate_threshold = ISSUE_DETECTION_CONFIG["duplicate_threshold"]
    duplicate_indices = []
    for i in range(len(sim_matrix)):
        if np.max(sim_matrix[i]) > duplicate_threshold:
            duplicate_indices.append(i)
    
    results = []
    
    # Process Label Issues
    for idx in issues_indices:
        img_path = dataset.files[idx]
        given_label_idx = dataset.labels[idx]
        given_label = dataset.classes[given_label_idx]
        predicted_label_idx = np.argmax(pred_probs[idx])
        predicted_label = dataset.classes[predicted_label_idx]
        
        conf = float(np.max(pred_probs[idx]))
        
        # Only include if confidence meets threshold
        if conf < ISSUE_DETECTION_CONFIG["min_confidence_for_relabel"]:
            continue
            
        # Calculate quality score and severity
        quality_score = calculate_quality_score(pred_probs, labels, features, idx, class_centroids)
        severity = classify_severity(quality_score)
        
        # Get feature space info
        feature_info = get_feature_space_info(features, labels, idx, num_classes)
        
        # Map similar indices to file paths
        similar_paths = [dataset.files[i] for i in feature_info["similar_samples_indices"][:3]]
        


        results.append({
            "file_path": img_path,
            "issue_type": "label_issue",
            "severity": severity,
            "quality_score": quality_score,
            "given_label": given_label,
            "suggested_label": predicted_label,
            "confidence": float(np.max(pred_probs[idx])),
            "split": split_name,
            "details": {
                "prediction_entropy": float(calculate_prediction_entropy(pred_probs[idx])),
                "feature_distance": feature_info["feature_distance"],
                "normalized_distance": feature_info["normalized_distance"],
                "similar_samples": similar_paths
            }
        })

    # Process Outliers
    for idx in outlier_indices:
        # Avoid duplicate entries if already a label issue
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
        
        # Quality score for outliers based on isolation
        feature_info = get_feature_space_info(features, labels, idx, num_classes)
        # Higher normalized distance = worse quality
        quality_score = min(feature_info["normalized_distance"] / 2.0, 1.0)  
        severity = classify_severity(quality_score)
        
        similar_paths = [dataset.files[i] for i in feature_info["similar_samples_indices"][:3]]
            
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "outlier",
            "severity": severity,
            "quality_score": quality_score,
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete",
            "confidence": 0.0,  # Outliers don't have prediction confidence
            "split": split_name,
            "details": {
                "prediction_entropy": 0.0,
                "feature_distance": feature_info["feature_distance"],
                "normalized_distance": feature_info["normalized_distance"],
                "similar_samples": similar_paths
            }
        })
        
    # Process Duplicates
    for idx in duplicate_indices:
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
        
        # Find the most similar image
        similarities = sim_matrix[idx]
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "duplicate",
            "severity": "LOW",  # Duplicates are usually low priority
            "quality_score": 0.3,  # Fixed low score for duplicates
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete",
            "confidence": float(similarity_score),
            "split": split_name,
            "details": {
                "prediction_entropy": 0.0,
                "feature_distance": 0.0,
                "normalized_distance": 0.0,
                "similar_samples": [dataset.files[most_similar_idx]]
            }
        })

    return results

def detect_issues():
    """
    Smart detection with comprehensive analysis and class-wise summary.
    Returns dict with issues list and class-wise statistics.
    
    Strategy:
    - Train Split: Use Feature Extraction + Cross-Validation (via detect_issues_in_split).
      Why? Using the trained model on training data introduces circular bias (it memorized the errors).
      CV ensures predictions are 'out-of-sample' relative to the classifier.
    - Val Split: Use the Best Trained Model (via detect_issues_with_model).
      Why? The model likely hasn't seen validation data, so its predictions are unbiased and 
      likely more accurate than generic features since it's fine-tuned.
    """
    best_model_path = os.path.join(MODELS_DIR, "best_model.pth")
    
    # Check if we have a trained model
    has_model = os.path.exists(best_model_path)
    
    print(f"Smart Analysis Strategy: {'Hybrid (Model + CV)' if has_model else 'CV-Only'}")
    
    all_issues = []
    strategy = "CV-Only (Resource Restricted)" if not has_model else f"Hybrid (Model: {os.path.basename(best_model_path)})"
    
    # 1. Analyze Training Data
    # STRATEGY: Use Cross-Validation for Training Data
    # Why? Dataset has ~30% noise. Using trained model (which memorized noise) is biased.
    # CV provides out-of-sample predictions to find real errors.
    print("Analyzing Training Split (using Cross-Validation context)...")
    train_issues = detect_issues_in_split("train", TRAIN_DIR)
    all_issues.extend(train_issues)
    
    # 2. Analyze Validation Data
    if has_model:
        print(f"Analyzing Validation Split (using Best Model at {os.path.basename(best_model_path)})...")
        # Use the trained model for validation as it's the 'expert'
        val_issues = detect_issues_with_model(best_model_path, "val", VAL_DIR)
        all_issues.extend(val_issues)
    else:
        print("Analyzing Validation Split (using Cross-Validation context)...")
        val_issues = detect_issues_in_split("val", VAL_DIR)
        all_issues.extend(val_issues)
    
    # 3. Analyze Test Data (if exists)
    if os.path.exists(TEST_DIR):
        if has_model:
            print("Analyzing Test Split (using Best Model)...")
            test_issues = detect_issues_with_model(best_model_path, "test", TEST_DIR)
        else:
            print("Analyzing Test Split (using Cross-Validation context)...")
            test_issues = detect_issues_in_split("test", TEST_DIR)
        all_issues.extend(test_issues)
    
    # 4. Generate Summary
    # Get class names from train or val dataset
    try:
        class_names = []
        if os.path.exists(TRAIN_DIR):
            dataset = CustomImageDataset(TRAIN_DIR)
            class_names = dataset.classes
        elif os.path.exists(VAL_DIR):
            dataset = CustomImageDataset(VAL_DIR)
            class_names = dataset.classes
    except:
        class_names = []
    
    if class_names:
        class_summary = get_class_issue_summary(all_issues, class_names)
        
        # Log summary table
        print("\n" + "="*80)
        print("CLASS-WISE ISSUE SUMMARY")
        print("="*80)
        print(f"{'Class':<20} {'Total':>8} {'Label':>8} {'Outlier':>8} {'Critical':>8} {'Avg Score':>10}")
        print("-"*80)
        for cls in class_names:
            stats = class_summary[cls]
            print(f"{cls:<20} {stats['total_issues']:>8} {stats['label_issues']:>8} "
                  f"{stats['outliers']:>8} {stats['critical_count']:>8} {stats['avg_quality_score']:>10.3f}")
        print("="*80 + "\n")
        
        return {
            "issues": all_issues,
            "class_summary": class_summary,
            "total_issues": len(all_issues),
            "strategy": strategy
        }
    else:
        return {"issues": all_issues, "class_summary": {}, "total_issues": len(all_issues), "strategy": strategy}

def detect_issues_with_model(model_path, split_name="train", split_dir=TRAIN_DIR):
    """
    Uses the trained model to find label issues with high precision (Enhanced Hybrid Approach).
    Includes ensemble predictions, quality scores, and detailed feature analysis.
    """
    if not HAS_CLEANLAB or not HAS_TORCH:
        return []
    
    if not os.path.exists(model_path):
        return []

    print(f"Loading model from {model_path} for Enhanced Hybrid Analysis on {split_name}...")
    
    # Load Data
    if not os.path.exists(split_dir):
        return []
        
    dataset = CustomImageDataset(split_dir, transform=val_transform)
    if len(dataset) == 0:
        return []
    
    if len(dataset) < 10:
        print(f"Dataset {split_name} too small for model-based detection.")
        return []
        
    loader = DataLoader(
        dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    
    # Load Model
    num_classes = len(dataset.classes)
    try:
        model = create_model(num_classes, "resnet18") 
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # Get model probabilities
    print(f"Getting model predictions on {split_name}...")
    all_probs_model = []
    all_labels = []
    
    import torch.nn.functional as F
    
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs_model.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_probs_model = np.vstack(all_probs_model)
    all_labels = np.array(all_labels)
    
    # Extract features for ensemble and quality scoring
    print(f"Extracting features for ensemble analysis on {split_name}...")
    features = extract_features(dataset)
    
    # Ensemble: Combine model predictions with feature-based classifier
    ensemble_probs = all_probs_model  # Start with model predictions
    use_ensemble = ISSUE_DETECTION_CONFIG.get("use_ensemble", True)
    
    if use_ensemble:
        print(f"Training ensemble classifier on {split_name}...")
        from sklearn.linear_model import LogisticRegression
        try:
            # Train a LogReg on features
            lr_clf = LogisticRegression(max_iter=1000, random_state=42)
            lr_clf.fit(features, all_labels)
            lr_probs = lr_clf.predict_proba(features)
            
            # Weighted ensemble calculation
            weights = ISSUE_DETECTION_CONFIG.get("ensemble_weights", {"model": 0.85, "aux": 0.15})
            ensemble_probs = weights["model"] * all_probs_model + weights["aux"] * lr_probs
            print(f"Ensemble created (Model {weights['model']} + LogReg {weights['aux']}) for {split_name}")
        except Exception as e:
            print(f"Ensemble training failed, using model-only: {e}")
    
    # Calculate class centroids
    class_centroids = calculate_class_centroids(features, all_labels, num_classes)
    
    # Cleanlab with ensemble probabilities
    print(f"Running Cleanlab with ensemble probabilities on {split_name}...")
    try:
        issues_indices = find_label_issues(
            labels=all_labels,
            pred_probs=ensemble_probs,
            return_indices_ranked_by="self_confidence"
        )
    except Exception as e:
        print(f"Cleanlab failed on {split_name}: {e}")
        return []
    
    results = []
    # Process Label Issues
    for idx in issues_indices:
        img_path = dataset.files[idx]
        given_label_idx = dataset.labels[idx]
        given_label = dataset.classes[given_label_idx]
        predicted_label_idx = np.argmax(ensemble_probs[idx])
        predicted_label = dataset.classes[predicted_label_idx]
        conf = float(np.max(ensemble_probs[idx]))
        
        # Only include if confidence meets threshold
        if conf < ISSUE_DETECTION_CONFIG["min_confidence_for_relabel"]:
            continue
            
        # Calculate quality score
        quality_score = calculate_quality_score(ensemble_probs, all_labels, features, idx, class_centroids)
        severity = classify_severity(quality_score)
        
        # Get feature space info
        feature_info = get_feature_space_info(features, all_labels, idx, num_classes)
        similar_paths = [dataset.files[i] for i in feature_info["similar_samples_indices"][:3]]
        
        # Calculate ensemble agreement (if ensemble was used)
        ensemble_agreement = 1.0
        if use_ensemble:
            model_pred = np.argmax(all_probs_model[idx])
            ensemble_pred = np.argmax(ensemble_probs[idx])
            ensemble_agreement = 1.0 if model_pred == ensemble_pred else 0.0

        results.append({
            "file_path": img_path,
            "issue_type": "hybrid_label_issue",
            "severity": severity,
            "quality_score": quality_score,
            "given_label": given_label,
            "suggested_label": predicted_label,
            "confidence": conf,
            "split": split_name,
            "details": {
                "prediction_entropy": float(calculate_prediction_entropy(ensemble_probs[idx])),
                "feature_distance": feature_info["feature_distance"],
                "normalized_distance": feature_info["normalized_distance"],
                "similar_samples": similar_paths,
                "ensemble_agreement": ensemble_agreement
            }
        })

    # 2. Outlier Detection (Ensemble)
    print(f"Finding outliers in {split_name} (ensemble method)...")
    outlier_indices = detect_outliers_ensemble(features, all_labels)
    
    for idx in outlier_indices:
        # Avoid duplicate entries if already a label issue
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
        
        # Quality score for outliers based on isolation
        feature_info = get_feature_space_info(features, all_labels, idx, num_classes)
        # Higher normalized distance = worse quality
        quality_score = min(feature_info["normalized_distance"] / 2.0, 1.0)  
        severity = classify_severity(quality_score)
        
        similar_paths = [dataset.files[i] for i in feature_info["similar_samples_indices"][:3]]
            
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "outlier",
            "severity": severity,
            "quality_score": quality_score,
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete",
            "confidence": 0.0,  # Outliers don't have prediction confidence
            "split": split_name,
            "details": {
                "prediction_entropy": 0.0,
                "feature_distance": feature_info["feature_distance"],
                "normalized_distance": feature_info["normalized_distance"],
                "similar_samples": similar_paths
            }
        })
    
    # 3. Near-Duplicate Detection
    print(f"Finding duplicates in {split_name}...")
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, 0)
    duplicate_threshold = ISSUE_DETECTION_CONFIG["duplicate_threshold"]
    duplicate_indices = []
    for i in range(len(sim_matrix)):
        if np.max(sim_matrix[i]) > duplicate_threshold:
            duplicate_indices.append(i)
            
    for idx in duplicate_indices:
        if any(r["file_path"] == dataset.files[idx] for r in results):
            continue
        
        # Find the most similar image
        similarities = sim_matrix[idx]
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        results.append({
            "file_path": dataset.files[idx],
            "issue_type": "duplicate",
            "severity": "LOW",  # Duplicates are usually low priority
            "quality_score": 0.3,  # Fixed low score for duplicates
            "given_label": dataset.classes[dataset.labels[idx]],
            "suggested_label": "delete",
            "confidence": float(similarity_score),
            "split": split_name,
            "details": {
                "prediction_entropy": 0.0,
                "feature_distance": 0.0,
                "normalized_distance": 0.0,
                "similar_samples": [dataset.files[most_similar_idx]]
            }
        })
        
    return results

def apply_fix(file_path, action, new_label=None):
    """
    Actions:
    - 'delete': Delete the file.
    - 'move': Move to new label folder (keeping same split).
    - 'ignore': Do nothing (mark as resolved in UI state).
    """
    if not os.path.exists(file_path):
        return False, "File not found"
        
    try:
        if action == 'delete':
            os.remove(file_path)
            return True, "Deleted"
        elif action == 'move' and new_label:
            file_name = os.path.basename(file_path)
            # Find the split directory (parent of parent of file)
            # file is at .../split/class/image.jpg
            # dir(file) = .../split/class
            # dir(dir(file)) = .../split
            
            current_class_dir = os.path.dirname(file_path)
            split_dir = os.path.dirname(current_class_dir)
            
            new_dir = os.path.join(split_dir, new_label)
            os.makedirs(new_dir, exist_ok=True)
            
            new_path = os.path.join(new_dir, file_name)
            shutil.move(file_path, new_path)
            return True, f"Moved to {new_label}"
    except Exception as e:
        return False, str(e)
    
    return True, "Ignored"
