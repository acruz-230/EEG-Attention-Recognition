import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet

# ---------------------------
# 1) Reproducibility / device
# ---------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2) Custom Dataset for MATLAB file
# ---------------------------
class MatSleepDataset(Dataset):
    def __init__(self, mat_path):
        self.mat_path = mat_path

        with h5py.File(mat_path, "r") as f:
            # X shape observed: (2650, 3000)
            X = np.array(f["X"], dtype=np.float32)

            # Ynum shape observed: (1, 2650)
            y = np.array(f["Ynum"]).squeeze()

            # Convert labels to integer class IDs
            y = y.astype(np.int64)

        # Sanity checks
        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D [epochs, samples], got shape {X.shape}")

        if len(X) != len(y):
            raise ValueError(f"Mismatch: X has {len(X)} epochs but y has {len(y)} labels")

        self.X = X
        self.y = y

        print("Loaded X shape:", self.X.shape)
        print("Loaded y shape:", self.y.shape)
        print("Unique labels:", np.unique(self.y, return_counts=True))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # shape: [3000]
        y = torch.tensor(self.y[idx], dtype=torch.long)

        # Since this dataset appears to be single-channel, reshape to [C, T] = [1, 3000]
        x = x.unsqueeze(0)

        return x, y

# ---------------------------
# 3) Load dataset
# ---------------------------
dataset = MatSleepDataset("./sleep_epoch_dataset.mat")

# ---------------------------
# 4) Train / validation split
# ---------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 5) Model
# ---------------------------
# IMPORTANT:
# Your MATLAB file appears to contain 1 channel per epoch, not 2.
# So num_electrodes should be 1 here.
model = EEGNet(
    chunk_size=3000,
    num_electrodes=1,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=5
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
            x = x.to(device).float()   # x shape from loader: [B, 1, 3000]
            y = y.to(device).long()

            # EEGNet expects [B, 1, C, T]
            x = x.unsqueeze(1)         # -> [B, 1, 1, 3000]

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ---------------------------
# 8) Training loop
# ---------------------------
num_epochs = 100
best_val_loss = float('inf')
patience = 2
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device).float()   # [B, 1, 3000]
        y = y.to(device).long()

        # reshape to [B, 1, C, T]
        x = x.unsqueeze(1)         # [B, 1, 1, 3000]

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

    # Early stopping check must happen AFTER val_loss is computed
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0

        # Save best model
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered")
        break