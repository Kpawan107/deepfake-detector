# cnn_detector/xception_for_inference.py

import torch
from timm import create_model
import torch.nn as nn

def xception(num_classes=2):
    """
    Returns an Xception model compatible with pretrained weights and inference.
    """
    model = create_model("xception", pretrained=False, num_classes=num_classes)
    
    # If your model was trained with a different classifier head, make sure this matches it
    if not isinstance(model.get_classifier(), nn.Linear) or model.get_classifier().out_features != num_classes:
        in_features = model.get_classifier().in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model
