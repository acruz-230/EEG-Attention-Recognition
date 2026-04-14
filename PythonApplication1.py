import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet

# ---------------------------
# 1) Reproducibility / device
# ---------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2) Dataset for MATLAB file
# ---------------------------
class MatlabSingleNodeEEGDataset(Dataset):
    def __init__(self, mat_path, x_key="X", y_key="Ynum"):
        self.mat_path = mat_path
        self.x_key = x_key
        self.y_key = y_key

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Could not find dataset file:\n{mat_path}")

        data = scipy.io.loadmat(mat_path)

        if x_key not in data:
            raise KeyError(f"'{x_key}' not found in MATLAB file. Keys: {list(data.keys())}")

        if y_key not in data:
            raise KeyError(f"'{y_key}' not found in MATLAB file. Keys: {list(data.keys())}")

        X = data[x_key]
        Y = data[y_key]

        print("Raw X shape from .mat:", X.shape)
        print("Raw Y shape from .mat:", Y.shape)

        # Expecting X = [epochs, time] = [293, 640]
        if X.ndim != 2:
            raise ValueError(
                f"Expected X to be 2D [epochs, time], but got shape {X.shape}"
            )

        # Flatten labels from [293,1] -> [293]
        Y = np.squeeze(Y)

        if Y.ndim != 1:
            raise ValueError(
                f"Expected Ynum to become 1D after squeeze, but got shape {Y.shape}"
            )

        if X.shape[0] != len(Y):
            raise ValueError(
                f"Mismatch: X has {X.shape[0]} epochs but Y has {len(Y)} labels"
            )

        # Convert to correct types
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)

        # Optional: if MATLAB labels start at 1 instead of 0, shift them down
        unique_labels = np.unique(self.Y)
        if np.min(unique_labels) == 1:
            print("Detected labels starting at 1. Converting to 0-based labels.")
            self.Y = self.Y - 1
            unique_labels = np.unique(self.Y)

        print("Loaded X shape:", self.X.shape)
        print("Loaded y shape:", self.Y.shape)
        print("Unique labels:", np.unique(self.Y, return_counts=True))

        self.num_epochs = self.X.shape[0]
        self.samples_per_epoch = self.X.shape[1]

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, idx):
        # x shape: [640]
        x = self.X[idx]

        # y shape: scalar
        y = self.Y[idx]

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 640]
        y = torch.tensor(y, dtype=torch.long)

        return x, y

# ---------------------------
# 3) Load dataset
# ---------------------------
mat_file = r"./AF3/p1_04_02_26_EPOCX_108311_2026.04.02T17.21.43.04.00_AF3_dataset.mat"

dataset = MatlabSingleNodeEEGDataset(
    mat_path=mat_file,
    x_key="X",
    y_key="Ynum"
)

# ---------------------------
# 4) Train / validation split
# ---------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")

# ---------------------------
# 5) Model
# ---------------------------
num_classes = 4


model = EEGNet(
    chunk_size=640,
    num_electrodes=1,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=4
).to(device)

print("\nModel created:")
print(model)

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
    best_val_loss = float("inf")
    best_epoch = -1
    best_train_loss = None
    best_train_acc = None
    best_val_acc = None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()   # [B, 1, 640]
            y = y.to(device).long()

            # EEGNet expects [B, 1, C, T]
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
num_epochs = 100
best_val_loss = float("inf")
patience = 15
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device).float()   # [B, 1, 640]
        y = y.to(device).long()

        # EEGNet expects [B, 1, C, T]
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
        best_epoch = epoch + 1

        best_train_loss = train_loss
        best_train_acc = train_acc
        best_val_acc = val_acc

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
print("\n===== BEST EPOCH RESULTS =====")
print(f"Best Epoch: {best_epoch}")
print(f"Train Loss: {best_train_loss:.4f}")
print(f"Train Acc : {best_train_acc:.4f}")
print(f"Val Loss  : {best_val_loss:.4f}")
print(f"Val Acc   : {best_val_acc:.4f}")
print("================================\n")
print("\n20 random predictions vs actual labels:\n")

num_to_show = min(20, len(dataset))
indices = random.sample(range(len(dataset)), num_to_show)

with torch.no_grad():
    for idx in indices:
        x, y = dataset[idx]

        x = x.unsqueeze(0)   # [1, 1, 640]
        x = x.unsqueeze(1)   # [1, 1, 1, 640]
        x = x.to(device).float()

        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

        print(f"Sample {idx}: Predicted = {pred}, Actual = {y.item()}")