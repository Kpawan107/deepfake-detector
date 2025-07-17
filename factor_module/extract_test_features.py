import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from timm import create_model

# ========== CONFIG ==========
TEST_DIR = "test_data/factor_test_aligned"
OUTPUT_EMB = "data/test.npy"
OUTPUT_LABELS = "data/labels.npy"
MODEL_NAME = "resnet18"
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

# ========== EXTRACT TEST FEATURES ==========
def extract_test_features():
    model = load_model()
    all_embeddings = []
    all_labels = []

    print(f"🔍 Scanning test images in {TEST_DIR}...")

    for label_name, label in [('real', 0), ('fake', 1)]:
        label_dir = os.path.join(TEST_DIR, label_name)
        if not os.path.exists(label_dir):
            print(f"⚠️ Directory not found: {label_dir}")
            continue

        images = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"📁 Found {len(images)} '{label_name}' images.")

        for img_name in tqdm(images, desc=f"🧠 Processing {label_name}"):
            img_path = os.path.join(label_dir, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = model(img_tensor)
                feat = feat.cpu().numpy().flatten()
                feat = feat / np.linalg.norm(feat)

                all_embeddings.append(feat)
                all_labels.append(label)

            except Exception as e:
                print(f"❌ Failed: {img_path} — {e}")

    # Save to disk
    np.save(OUTPUT_EMB, np.array(all_embeddings))
    np.save(OUTPUT_LABELS, np.array(all_labels))
    print(f"\n✅ Saved features to {OUTPUT_EMB}")
    print(f"📝 Saved labels to {OUTPUT_LABELS}")

# ========== MAIN ==========
if __name__ == "__main__":
    extract_test_features()
