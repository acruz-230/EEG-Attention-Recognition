
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet
import random

# ---------------------------
# 1) Reproducibility / device
# ---------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2) Custom Dataset (CSV epochs)
# ---------------------------
class SingleNodeEEGDataset(Dataset):
    def __init__(self, data_dir, node_name="AF3", label_map=None):
        self.data_dir = data_dir
        self.node_name = node_name

        if label_map is None:
            label_map = {
                "NoMusic_Question": 0,
                "Music_Question": 1
            }
        self.label_map = label_map

        # Load all CSV files
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        if len(self.file_list) == 0:
            raise ValueError(f"No CSV files found in {data_dir}")

        print(f"Found {len(self.file_list)} epoch files.")

        # Inspect first file
        df = pd.read_csv(self.file_list[0])

        if self.node_name not in df.columns:
            raise ValueError(f"Column '{self.node_name}' not found.")

        if "label" not in df.columns:
            raise ValueError("Column 'label' not found.")

        self.samples_per_epoch = len(df)

        print("Loaded X shape:", (len(self.file_list), self.samples_per_epoch))
        print("Loaded y shape:", (len(self.file_list),))

        # Collect labels
        labels = []
        for f in self.file_list:
            d = pd.read_csv(f)
            raw_label = d["label"].iloc[0]

            if raw_label not in self.label_map:
                raise ValueError(f"Unknown label '{raw_label}' in {f}")

            labels.append(self.label_map[raw_label])

        labels = np.array(labels)

        print("Unique labels:", np.unique(labels, return_counts=True))

        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        df = pd.read_csv(self.file_list[idx])

        # Signal: [640]
        x = df[self.node_name].to_numpy(dtype=np.float32)

        # Label
        y = self.labels[idx]

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 640]
        y = torch.tensor(y, dtype=torch.long)

        return x, y

# ---------------------------
# 3) Load dataset
# ---------------------------
dataset = SingleNodeEEGDataset("./AF3", node_name="AF3")

# ---------------------------
# 4) Train / validation split
# ---------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ---------------------------
# 5) Model
# ---------------------------
model = EEGNet(
    chunk_size=640,
    num_electrodes=1,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=2
).to(device)

# ---------------------------
# 6) Loss / optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 7) Evaluation function
# ---------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()   # [B, 1, 640]
            y = y.to(device).long()

            x = x.unsqueeze(1)         # [B, 1, 1, 640]

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

# ---------------------------
# 8) Training loop
# ---------------------------
num_epochs = 50
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device).float()   # [B, 1, 640]
        y = y.to(device).long()

        x = x.unsqueeze(1)         # [B, 1, 1, 640]

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break

# ---------------------------
# 9) Load best model
# ---------------------------
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ---------------------------
# 10) Print predictions
# ---------------------------
print("\n20 random predictions vs actual labels:\n")

indices = random.sample(range(len(dataset)), 20)

with torch.no_grad():
    for idx in indices:
        x, y = dataset[idx]

        x = x.unsqueeze(0)   # [1, 1, 640]
        x = x.unsqueeze(1)   # [1, 1, 1, 640]
        x = x.to(device).float()

        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

        print(f"Sample {idx}: Predicted = {pred}, Actual = {y.item()}")
