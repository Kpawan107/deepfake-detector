import os
import timm
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "models/xception_focal_best.pth"
TEST_DIR = "test_data/preprocessed_xception/celeb_df_v2"
BATCH_SIZE = 64
IMG_SIZE = 299

def custom_label(path):
    path = path.lower()
    return 0 if ("real" in path or "original" in path) else 1

class RealFakeTestDataset(ImageFolder):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        label = custom_label(self.imgs[index][0])
        return img, label

def plot_confusion_matrix(cm, labels, path="confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    print(f"🖼️ Confusion matrix saved: {path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    print(f"🖼️ ROC curve saved: {path}")
    plt.close()

def main():
    print("📂 Loading test data...")
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = RealFakeTestDataset(TEST_DIR, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("📦 Loading XceptionNet model...")
    model = timm.create_model('xception', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.cuda().eval()

    all_preds, all_labels, all_probs = [], [], []

    print("🧠 Running inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.cuda()
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.extend(probs[:, 1])  # probability of fake
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(MODEL_PATH)
    print(TEST_DIR)

    print("\n✅ Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    cm = confusion_matrix(all_labels, all_preds)
    print("📊 Confusion Matrix:\n", cm)

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"📈 ROC-AUC Score: {auc:.4f}")
        plot_confusion_matrix(cm, labels=["Real", "Fake"])
        plot_roc_curve(all_labels, all_probs)
    except:
        print("⚠️ Could not compute ROC-AUC or plot.")

if __name__ == "__main__":
    main()
