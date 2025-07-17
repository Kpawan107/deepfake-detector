import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from timm import create_model

# ==================== CONFIG ======================
REFERENCE_DIR = "data/reference_images"
OUTPUT_EMB = "data/reference_embeddings.npy"
OUTPUT_LABELS = "data/reference_labels.npy"
MODEL_NAME = "resnet18"  # or use iresnet50 / iresnet100 from FaceX-Zoo
IMG_SIZE = 112
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== TRANSFORM ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ==================== MODEL ======================
def load_model():
    model = create_model(MODEL_NAME, pretrained=True, num_classes=512)
    model.eval()
    return model.to(DEVICE)

# ==================== MAIN ======================
def extract_reference_embeddings():
    model = load_model()
    identity_embeddings = []
    identity_labels = []

    identities = sorted([d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))])
    for identity in tqdm(identities, desc="Extracting features"):
        identity_dir = os.path.join(REFERENCE_DIR, identity)
        images = [f for f in os.listdir(identity_dir) if f.endswith(('.jpg', '.png'))]
        features = []

        for img_name in images:
            img_path = os.path.join(identity_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = model(img_tensor)
                feat = feat.cpu().numpy().flatten()
                feat = feat / np.linalg.norm(feat)
                features.append(feat)
            except Exception as e:
                print(f"❌ Failed to process {img_path}: {e}")

        if len(features) > 0:
            mean_feat = np.mean(features, axis=0)
            identity_embeddings.append(mean_feat)
            identity_labels.append(identity)
        else:
            print(f"⚠️ Skipped {identity}: no valid features extracted.")

    # Save to disk
    np.save(OUTPUT_EMB, np.array(identity_embeddings))
    np.save(OUTPUT_LABELS, np.array(identity_labels))
    print(f"✅ Saved {len(identity_embeddings)} reference embeddings to {OUTPUT_EMB}")

if __name__ == "__main__":
    extract_reference_embeddings()
