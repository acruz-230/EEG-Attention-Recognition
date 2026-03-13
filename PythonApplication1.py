import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torcheeg.datasets import SleepEDFxDataset
from torcheeg import transforms
from torcheeg.models import EEGNet

# ---------------------------
# 1) Reproducibility / device
# ---------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 2) Dataset
# ---------------------------

label_map = {
    "W": 0,   # Wake
    "1": 1,   # N1
    "2": 2,   # N2
    "3": 3,   # N3
    "4": 4,   # N4
    "R": 5    # REM
}
dataset = SleepEDFxDataset(
    root_path="./sleep-edf-database-expanded-1.0.0",
    version="cassette",
    num_channel=2,
    chunk_size=3000,
    remove_unclear_example=True,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select(key="label"),
        transforms.Lambda(lambda s: 0 if ("W" in s or s == "W") else 1)
    ])
)
print("Total samples:", len(dataset))

# ---------------------------
# 3) Train / validation split
# ---------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ---------------------------
# 4) Model
# ---------------------------
model = EEGNet(
    chunk_size=3000,
    num_electrodes=2,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=6
).to(device)

# ---------------------------
# 5) Loss / optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 6) Evaluation function
# ---------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()

            # EEGNet expects [B, 1, C, T]
            x = x.unsqueeze(1)

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
# 7) Training loop
# ---------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device).float()
        y = y.to(device).long()

        # reshape to [B, 1, C, T]
        x = x.unsqueeze(1)

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
