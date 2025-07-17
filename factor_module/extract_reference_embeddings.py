import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from timm import create_model

# ========== CONFIG ==========
REFERENCE_DIR = "data/reference_aligned_faces"
OUTPUT_EMB = "data/reference_embeddings.npy"
OUTPUT_LABELS = "data/reference_labels.npy"
MODEL_NAME = "resnet18"  # or use iresnet50 for more powerful model
IMG_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== LOAD MODEL ==========
def load_model():
    model = create_model(MODEL_NAME, pretrained=True, num_classes=512)
    model.eval()
    return model.to(DEVICE)

# ========== EXTRACT EMBEDDINGS ==========
def extract_embeddings():
    model = load_model()
    identity_embeddings = []
    identity_labels = []

    all_files = [f for f in os.listdir(REFERENCE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 Extracting reference embeddings for {len(all_files)} images...")

    for file_name in tqdm(all_files):
        img_path = os.path.join(REFERENCE_DIR, file_name)
        identity_name = os.path.splitext(file_name)[0]  # e.g. Barack_Obama

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                feat = model(img_tensor)
            feat = feat.cpu().numpy().flatten()
            feat = feat / np.linalg.norm(feat)

            identity_embeddings.append(feat)
            identity_labels.append(identity_name)

        except Exception as e:
            print(f"⚠️ Failed to process {img_path}: {e}")

    # Save
    np.save(OUTPUT_EMB, np.array(identity_embeddings))
    np.save(OUTPUT_LABELS, np.array(identity_labels))
    print(f"✅ Saved {len(identity_embeddings)} reference embeddings to {OUTPUT_EMB}")
    print(f"📝 Corresponding labels saved to {OUTPUT_LABELS}")

# ========== MAIN ==========
if __name__ == "__main__":
    extract_embeddings()
