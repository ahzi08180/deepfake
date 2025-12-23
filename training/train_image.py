import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import RVF10KDataset

# =====================
# Config
# =====================
DATASET_ROOT = "../data/rvf10k"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

MODEL_SAVE_PATH = "../saved_models/deepfake_model.pth"
RESULT_DIR = "../results"
os.makedirs(RESULT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Dataset
# =====================
train_ds = RVF10KDataset(DATASET_ROOT, "train")
val_ds = RVF10KDataset(DATASET_ROOT, "valid")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# Model
# =====================
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(1280, 1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# Training
# =====================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

# =====================
# Save Model
# =====================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# =====================
# Evaluation (Validation)
# =====================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(labels.cpu().numpy())

# =====================
# Confusion Matrix
# =====================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',        # 🔵 藍色主題
        cbar=True,
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

plot_confusion_matrix(all_labels, all_preds, os.path.join(RESULT_DIR, "confusion_matrix.png"))

# =====================
# Classification Report
# =====================
report = classification_report(
    all_labels,
    all_preds,
    target_names=["Real", "Fake"],
    digits=4
)

print("\nClassification Report:")
print(report)

with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("\n📁 Evaluation results saved in 'results/'")
