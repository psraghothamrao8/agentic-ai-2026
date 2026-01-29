
import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes, model_name="resnet18", pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(weights="DEFAULT" if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
         model = models.resnet50(weights="DEFAULT" if pretrained else None)
         num_ftrs = model.fc.in_features
         model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights="DEFAULT" if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model
