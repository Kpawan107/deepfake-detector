# factor_module/factor.py

import os
import torch
import numpy as np
from timm import create_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ========== CONFIG ==========
MODEL_NAME = "resnet18"
REFERENCE_EMB_PATH = "data/reference_embeddings.npy"
REFERENCE_LABELS_PATH = "data/reference_labels.npy"
IMG_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== LOAD REFERENCE EMBEDDINGS ==========
reference_embeddings = np.load(REFERENCE_EMB_PATH)
reference_labels = np.load(REFERENCE_LABELS_PATH)

# ========== LOAD MODEL ==========
model = create_model(MODEL_NAME, pretrained=True, num_classes=512)
model.eval()
model = model.to(DEVICE)

# ========== MAIN FUNCTION ==========
def compute_factor_score(pil_image):
    """
    Takes PIL image and returns similarity score to closest reference identity.
    """
    try:
        img = transform(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(img)
        feat = feat.cpu().numpy().flatten()
        feat = feat / np.linalg.norm(feat)

        # Cosine similarity with all reference embeddings
        similarities = np.dot(reference_embeddings, feat)
        max_sim = np.max(similarities)

        return float(max_sim)  # Return similarity as score (0 to 1)
    except Exception as e:
        print(f"⚠️ compute_factor_score error: {e}")
        return 0.0
