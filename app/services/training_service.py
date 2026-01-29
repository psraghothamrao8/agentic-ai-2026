import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader

try:
    import optuna
except ImportError:
    pass

from app.config import MODELS_DIR, LOGS_DIR, TRAIN_DIR, VAL_DIR
from app.logging import setup_logger
from app.services.data_service import CustomImageDataset, train_transform, val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("pluto_trainer", os.path.join(LOGS_DIR, "pluto.log"), mode='w')

def create_model(num_classes):
    # Standard ResNet18 for this project
    model = models.resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    r_loss, correct, total = 0.0, 0, 0
    for img, lbl, _ in loader:
        img, lbl = img.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        r_loss += loss.item() * img.size(0)
        _, pred = torch.max(out, 1)
        total += lbl.size(0)
        correct += (pred == lbl).sum().item()
    return (r_loss / total, correct / total) if total > 0 else (0.0, 0.0)

def validate(model, loader, criterion):
    model.eval()
    r_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for img, lbl, _ in loader:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out = model(img)
            r_loss += criterion(out, lbl).item() * img.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    return (r_loss / total, correct / total) if total > 0 else (0.0, 0.0)

def train_core(params, ds_train, ds_val, epochs=300, min_epochs=30, patience=15):
    """Core training loop with Early Stopping."""
    model = create_model(len(ds_train.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 0.001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    
    train_loader = DataLoader(ds_train, batch_size=params.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=params.get("batch_size", 32), shuffle=False)
    
    best_loss = float('inf')
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    triggers = 0
    
    for epoch in range(epochs):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc = validate(model, val_loader, criterion)
        scheduler.step(v_loss)
        
        # Track best model (based on Loss for stability, or Acc for metric)
        # Using Loss for Early Stopping is standard
        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = v_acc # Keep corresponding acc
            best_wts = copy.deepcopy(model.state_dict())
            triggers = 0
        else:
            triggers += 1
            
        # Logging check
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train: {t_loss:.4f} | Val: {v_loss:.4f} (Acc: {v_acc:.2%}) | Patience: {triggers}/{patience}")

        # Early Stopping
        if triggers >= patience and (epoch + 1) >= min_epochs:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_wts)
    return model, best_acc

def run_automated_training(full_epochs=300):
    logger.info("Starting Auto-ML Training...")
    if not os.path.exists(TRAIN_DIR): return {"status": "error", "message": "No train dir"}
    
    ds_train = CustomImageDataset(TRAIN_DIR, transform=train_transform)
    ds_val = CustomImageDataset(VAL_DIR, transform=val_transform)
    
    # 1. Hyperparameter Search (Optuna)
    best_params = {"lr": 0.001, "batch_size": 32}
    
    try:
        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [16, 32])
            
            # Use larger subset for difficult datasets
            subset_size = min(len(ds_train), 1500)
            subset = torch.utils.data.Subset(ds_train, range(subset_size))
            
            # Increased trial epochs to 20 per user request
            _, acc = train_core({"lr": lr, "batch_size": bs}, subset, ds_val, epochs=20, min_epochs=10, patience=5)
            return acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)
        best_params = study.best_params
        logger.info(f"Best Params Found: {best_params}")
    except Exception as e:
        logger.warning(f"Optuna search failed/skipped: {e}. Using defaults.")

    # 2. Final Training
    # Max 300, Min 30, Patience 20 (robust)
    model, val_acc = train_core(best_params, ds_train, ds_val, epochs=full_epochs, min_epochs=30, patience=20)
    
    save_path = os.path.join(MODELS_DIR, "best_model.pth")
    tmp_path = save_path + ".tmp"
    torch.save(model.state_dict(), tmp_path)
    os.replace(tmp_path, save_path)
    
    return {"status": "completed", "accuracy": val_acc, "params": best_params}
