import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import seaborn as sns

# Load predicted.json
with open("predicted.json", "r") as f:
    data = json.load(f)

y_true = []
y_pred = []

# Normalize labels and collect
for entry in data:
    true_label = entry["ground_truth"].strip().lower()
    predicted_label = entry["prediction"].strip().lower()

    y_true.append(true_label)
    y_pred.append(predicted_label)

# Convert string labels to binary format for ROC-AUC
# Assuming "real" = 1, "fake" = 0
label_map = {"real": 1, "fake": 0}
y_true_bin = [label_map[label] for label in y_true]
y_pred_bin = [label_map[label] for label in y_pred]

# ----------------------------
# 🧮 Evaluation Metrics
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="real", zero_division=0)
recall = recall_score(y_true, y_pred, pos_label="real", zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label="real", zero_division=0)
roc_auc = roc_auc_score(y_true_bin, y_pred_bin)

print("\n📊 Evaluation Metrics:")
print(f"✅ Accuracy:        {accuracy:.4f}")
print(f"🎯 Precision:       {precision:.4f}")
print(f"📥 Recall:          {recall:.4f}")
print(f"🏆 F1 Score:        {f1:.4f}")
print(f"📈 ROC-AUC Score:   {roc_auc:.4f}")

# ----------------------------
# 📊 Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred, labels=["real", "fake"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ----------------------------
# 📄 Classification Report
# ----------------------------
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["real", "fake"]))

# ----------------------------
# 📉 ROC Curve
# ----------------------------
fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
