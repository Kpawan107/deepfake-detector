# train_xception_optuna.py (Optimized for RTX 3050: 4GB VRAM)

import os
import timm
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== CONFIG ==========
DATA_DIR = "data/aligned_faces"
IMG_SIZE = 299
BATCH_SIZE = 16  # ⚡ Optimized for 4GB VRAM
EPOCHS = 12
VAL_SPLIT = 0.15
SAVE_PATH = "models/xception_best.pth"

# ========== TRANSFORMS ==========
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # ⚡ Avoid extra resize & crop
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return train_tf, test_tf

# ========== DATASET ==========
def label_map(path):
    path = path.lower()
    return 0 if ("original" in path or "/real" in path or "\\real" in path) else 1

class RealFakeDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        label = label_map(self.imgs[index][0])
        return img, label

# ========== TRAINING FUNCTION ==========
def train_model(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    print(f"\n🔍 Starting trial with lr={lr:.5e}, weight_decay={wd:.5e}")

    train_tf, test_tf = get_transforms()
    full_ds = RealFakeDataset(DATA_DIR, transform=train_tf)
    val_len = int(len(full_ds) * VAL_SPLIT)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    val_ds.dataset.transform = test_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    model = timm.create_model('xception', pretrained=True, num_classes=2).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        print(f"🔧 Train Loss: {avg_loss:.4f}")

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x = x.cuda()
                outputs = model(x)
                preds = outputs.argmax(dim=1).cpu()
                y_true.extend(y)
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Val Accuracy: {acc*100:.2f}%")

        trial.report(acc, epoch)
        if trial.should_prune():
            print("⛔ Trial pruned")
            raise optuna.exceptions.TrialPruned()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Best model saved (Val Acc: {acc*100:.2f}%)")

    return best_acc

# ========== RUN OPTUNA TUNING ==========
if __name__ == '__main__':
    print("🔧 Starting Optuna tuning (resume enabled)")
    study = optuna.create_study(
        study_name="xception_tune",
        storage="sqlite:///cnn/xception_optuna.db",
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(train_model, n_trials=2)

    print("\n✅ Best Accuracy:", study.best_value)
    print("🎯 Best Hyperparameters:", study.best_params)
    print(f"📦 Best model saved to: {SAVE_PATH}")