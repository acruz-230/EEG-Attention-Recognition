import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet


# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")
print("Using device:", device)


# Dataset
class EEGDataset(Dataset):
    def __init__(self, data_path, Eeg_epochs="X", Epoch_labels="Ynum"):
        self.file_path = data_path
        self.x_data = Eeg_epochs
        self.y_labels = Epoch_labels

        #Error Checking/ File Sanity Checks
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find dataset file:\n{data_path}")

        data = scipy.io.loadmat(data_path)

        if Eeg_epochs not in data:
            raise KeyError(f"'{Eeg_epochs}' not found in MATLAB file. Keys: {list(data.keys())}")

        if Epoch_labels not in data:
            raise KeyError(f"'{Epoch_labels}' not found in MATLAB file. Keys: {list(data.keys())}")

        X = data[Eeg_epochs]
        Y = data[Epoch_labels]

        print("Raw X shape from .mat:", X.shape)
        print("Raw Y shape from .mat:", Y.shape)

        # [epochs, time] = [293, 640]
        if X.ndim != 2:
            raise ValueError(
                f"Expected X to be 2D [epochs, time], but got shape {X.shape}"
            )

        # [293]
        Y = np.squeeze(Y)

        if Y.ndim != 1:
            raise ValueError(
                f"Expected Ynum to become 1D after squeeze, but got shape {Y.shape}"
            )

        if X.shape[0] != len(Y):
            raise ValueError(
                f"Mismatch: X has {X.shape[0]} epochs but Y has {len(Y)} labels"
            )

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)

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
        x = self.X[idx]

        y = self.Y[idx]

        # [1, 640]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   
        y = torch.tensor(y, dtype=torch.long)

        return x, y


mat_file = r"./AF3/AF3_combined.mat"

dataset = EEGDataset(
    data_path=mat_file,
    Eeg_epochs="X",
    Epoch_labels="Ynum"
)


# Training loop definition
# 80% to train rest to validate
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

# EEGNet model definition
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

# Loss / optimizer definition
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)


#Testing Definition
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
            x = x.to(device).float()  
            y = y.to(device).long()

            x = x.unsqueeze(1)         

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total

# Training loop
loops = 100
best_val_loss = float("inf")
patience = 20
counter = 0

for epoch in range(loops):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_eeg, y_labels in train_loader:
        x_eeg = x_eeg.to(device).float()  
        y_labels = y_labels.to(device).long()

        x_eeg = x_eeg.unsqueeze(1)      

        optimizer.zero_grad()
        scores = model(x_eeg)
        loss = criterion(scores, y_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_eeg.size(0)
        preds = torch.argmax(scores, dim=1)
        correct += (preds == y_labels).sum().item()
        total += y_labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(
        f"Epoch [{epoch+1}/{loops}] | "
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

#Load best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


# RESULTS
print("\n__RESULTS__")
print(f"Best Epoch: {best_epoch}")
print(f"Train Loss: {best_train_loss:.4f}")
print(f"Train Acc : {best_train_acc:.4f}")
print(f"Val Loss  : {best_val_loss:.2f}")
print(f"Val Acc   : {best_val_acc:.2f}")
print("\n20 random predictions vs actual labels:\n")


#Print random samples with guess and results from trained model
rand_samp = 20
indices = random.sample(range(len(dataset)), rand_samp)

with torch.no_grad():
    for i in indices:
        x_eeg, y_labels = dataset[i]

        #conform to [B, 1, C, T]
        x_eeg = x_eeg.unsqueeze(0)   
        x_eeg = x_eeg.unsqueeze(1) 
        x_eeg = x_eeg.to(device).float()

        scores = model(x_eeg)
        score_prob = torch.softmax(scores, dim=1)   
        guess = torch.argmax(score_prob, dim=1).item()
        guess_prob = score_prob[0, guess].item()     


        print(f"Sample {i}: Predicted = {guess},{guess_prob:.2f}, Actual = {y_labels.item()}")