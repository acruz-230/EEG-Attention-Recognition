import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.io import loadmat 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1) Reproducibility / device
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")
print("Using device:", device)

# ============================================================
# 2) Settings
# ============================================================
MAT_PATH = "./AF3/AF3_combined.mat"   # change if needed
FS = 100                                 # change to your real sampling rate if different
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATIENCE = 15
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional label names if your labels are 0,1,2,3
LABEL_NAMES = {
    0: "NoMusic_NoQuestion",
    1: "Music_NoQuestion",
    2: "NoMusic_Question",
    3: "Music_Question"
}

# ============================================================
# 3) Dataset
# ============================================================
class MatEEGDataset(Dataset):
    def __init__(self, mat_path):
        self.mat_path = mat_path

        data = loadmat(mat_path)
        X = np.array(data["X"], dtype=np.float32)
        y = np.array(data["Ynum"]).squeeze().astype(np.int64)

        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D [epochs, samples], got shape {X.shape}")

        if len(X) != len(y):
            raise ValueError(f"Mismatch: X has {len(X)} epochs but y has {len(y)} labels")

        self.X = X
        self.y = y

        self.unique_labels = np.unique(self.y)
        self.num_classes = len(self.unique_labels)

        print("Loaded X shape:", self.X.shape)
        print("Loaded y shape:", self.y.shape)
        print("Unique labels:", np.unique(self.y, return_counts=True))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)   # [T]

        # normalize this epoch
        x = (x - x.mean()) / (x.std() + 1e-8)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        # single channel -> [C, T] = [1, T]
        x = x.unsqueeze(0)
        return x, y

dataset = MatEEGDataset(MAT_PATH)

# ============================================================
# 4) Train / validation split
# ============================================================
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

indices = np.arange(len(dataset))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=dataset.y
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 5) Model
# ============================================================
model = EEGNet(
    chunk_size=dataset.X.shape[1],   # e.g. 3000
    num_electrodes=1,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=dataset.num_classes
).to(device)

# ============================================================
# 6) Loss / optimizer
# ============================================================

class_counts = np.array([(dataset.y == c).sum() for c in dataset.unique_labels], dtype=np.float32)
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============================================================
# 7) Bandpower helpers
# ============================================================
def compute_bandpower(epoch, fs=100):
    """
    epoch: 1D numpy array [T]
    """
    freqs, psd = welch(epoch, fs=fs, nperseg=min(256, len(epoch)))

    def bp(fmin, fmax):
        idx = (freqs >= fmin) & (freqs <= fmax)
        if np.sum(idx) < 2:
            return 0.0
        return np.trapz(psd[idx], freqs[idx])

    return {
        "delta": bp(0.5, 4),
        "theta": bp(4, 8),
        "alpha": bp(8, 12),
        "beta": bp(12, 30),
    }

def bandpower_string(bp_dict):
    return ", ".join([f"{k}={v:.4f}" for k, v in bp_dict.items()])

# ============================================================
# 8) Evaluation
# ============================================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()          # [B, 1, T]
            y = y.to(device).long()
            x = x.unsqueeze(1)                # [B, 1, 1, T]

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

# ============================================================
# 9) Training loop
# ============================================================
best_val_loss = float("inf")
counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device).float()      # [B, 1, T]
        y = y.to(device).long()
        x = x.unsqueeze(1)            # [B, 1, 1, T]

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
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1

    if counter >= PATIENCE:
        print("Early stopping triggered")
        break

# Load best model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()



def evaluate_detailed(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()      # [B, 1, T]
            y = y.to(device).long()
            x = x.unsqueeze(1)            # [B, 1, 1, T]

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    print("\n===== DETAILED VALIDATION RESULTS =====")
    print("Predicted class counts:", np.unique(all_preds, return_counts=True))
    print("True class counts     :", np.unique(all_true, return_counts=True))
    print("\nConfusion Matrix:\n", confusion_matrix(all_true, all_preds))
    print("\nClassification Report:\n", classification_report(all_true, all_preds, digits=4))

# ============================================================
# 10) Prediction helper
# ============================================================
def predict_single_epoch(model, x):
    """
    x: torch tensor [1, T]
    """
    model.eval()
    with torch.no_grad():
        x_model = x.unsqueeze(0).unsqueeze(1).to(device).float()   # [1,1,1,T]
        logits = model(x_model)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs

# ============================================================
# 11) Saliency / influence analysis
# ============================================================
def compute_saliency(model, x, target_class=None):
    """
    x: torch tensor [1, T]
    Returns:
        saliency: numpy array [T]
        pred_class: int
        used_class: int
        logits_np: numpy array [num_classes]
    """
    model.eval()

    x_in = x.unsqueeze(0).unsqueeze(1).to(device).float()   # [1,1,1,T]
    x_in.requires_grad_(True)

    logits = model(x_in)
    pred_class = int(torch.argmax(logits, dim=1).item())

    if target_class is None:
        used_class = pred_class
    else:
        used_class = int(target_class)

    score = logits[0, used_class]
    model.zero_grad()
    score.backward()

    saliency = x_in.grad.detach().abs().cpu().numpy().squeeze()
    logits_np = logits.detach().cpu().numpy().squeeze()

    return saliency, pred_class, used_class, logits_np

def top_saliency_region(saliency, window=200):
    """
    Finds the most influential time region by averaging saliency in a moving window.
    """
    if len(saliency) < window:
        return 0, len(saliency)

    kernel = np.ones(window) / window
    smoothed = np.convolve(saliency, kernel, mode="valid")
    start = int(np.argmax(smoothed))
    end = start + window
    return start, end

def save_saliency_plot(epoch_signal, saliency, sample_idx, pred_class, actual_class, out_dir):
    t = np.arange(len(epoch_signal))

    plt.figure(figsize=(12, 5))
    plt.plot(t, epoch_signal, label="EEG signal")
    plt.plot(t, saliency / (np.max(saliency) + 1e-8) * np.std(epoch_signal), label="Normalized saliency")
    plt.title(f"Sample {sample_idx} | Pred={pred_class} | Actual={actual_class}")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude / relative saliency")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"saliency_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

# ============================================================
# 12) Optimize prototype signal for each class
# ============================================================
def total_variation_loss(x):
    return torch.mean(torch.abs(x[..., 1:] - x[..., :-1]))

def optimize_class_prototype(
    model,
    target_class,
    signal_len,
    steps=300,
    lr=0.01,
    l2_weight=1e-4,
    tv_weight=1e-4
):
    """
    Creates an input signal that maximizes one target class score.
    Returns:
        prototype_signal: numpy array [T]
        probs: numpy array [num_classes]
    """
    model.eval()

    x_opt = torch.randn((1, 1, 1, signal_len), device=device, requires_grad=True)
    opt = optim.Adam([x_opt], lr=lr)

    for step in range(steps):
        opt.zero_grad()

        logits = model(x_opt)
        class_score = logits[0, target_class]

        l2 = torch.mean(x_opt ** 2)
        tv = total_variation_loss(x_opt)

        # maximize class score while keeping signal smoother / smaller
        loss = -class_score + l2_weight * l2 + tv_weight * tv
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(x_opt)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        proto = x_opt.detach().cpu().numpy().squeeze()

    return proto, probs

def save_prototype_plot(signal, class_idx, out_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(f"Optimized Prototype Signal for Class {class_idx}")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"prototype_class_{class_idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def save_psd_plot(signal, fs, class_idx, out_dir):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd)
    plt.title(f"Prototype PSD for Class {class_idx}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"prototype_psd_class_{class_idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

# ============================================================
# 13) Print random predictions + bandpower
# ============================================================
print("\n===== RANDOM PREDICTIONS + BANDPOWER =====\n")
num_examples = min(10, len(dataset))
indices = random.sample(range(len(dataset)), num_examples)

for idx in indices:
    x, y = dataset[idx]
    epoch_np = x.squeeze().numpy()
    bp = compute_bandpower(epoch_np, fs=FS)
    pred, probs = predict_single_epoch(model, x)

    actual_name = LABEL_NAMES.get(int(y.item()), str(int(y.item())))
    pred_name = LABEL_NAMES.get(int(pred), str(int(pred)))

    print(f"Sample {idx}")
    print(f"  Actual    : {y.item()} ({actual_name})")
    print(f"  Predicted : {pred} ({pred_name})")
    print(f"  Probabilities: {np.round(probs, 4)}")
    print(f"  Bandpower : {bandpower_string(bp)}")
    print(f"  Bandpower : {bandpower_string(bp)}")
    print("-" * 60)

# ============================================================
# 14) saliency on a few random samples
# ============================================================
print("\n===== SALIENCY / MODEL INFLUENCE =====\n")

saliency_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

for idx in saliency_indices:
    x, y = dataset[idx]
    epoch_np = x.squeeze().numpy()

    saliency, pred_class, used_class, logits_np = compute_saliency(model, x)
    start, end = top_saliency_region(saliency, window=200)

    influential_segment = epoch_np[start:end]
    influential_bp = compute_bandpower(influential_segment, fs=FS)

    plot_path = save_saliency_plot(
        epoch_signal=epoch_np,
        saliency=saliency,
        sample_idx=idx,
        pred_class=pred_class,
        actual_class=int(y.item()),
        out_dir=OUTPUT_DIR
    )

    print(f"Sample {idx}")
    print(f"  Actual class         : {y.item()} ({LABEL_NAMES.get(int(y.item()), y.item())})")
    print(f"  Predicted class      : {pred_class} ({LABEL_NAMES.get(pred_class, pred_class)})")
    print(f"  Used class for grad  : {used_class}")
    print(f"  Logits               : {np.round(logits_np, 4)}")
    print(f"  Most influential region: samples {start} to {end}")
    print(f"  Influential region bandpower: {bandpower_string(influential_bp)}")
    print(f"  Saved saliency plot  : {plot_path}")
    print("-" * 60)

# ============================================================
# 15) optimize a prototype signal for each class
# ============================================================
print("\n===== CLASS PROTOTYPE SIGNALS =====\n")

signal_len = dataset.X.shape[1]

for class_idx in range(dataset.num_classes):
    proto_signal, proto_probs = optimize_class_prototype(
        model=model,
        target_class=class_idx,
        signal_len=signal_len,
        steps=300,
        lr=0.01,
        l2_weight=1e-4,
        tv_weight=1e-4
    )

    proto_bp = compute_bandpower(proto_signal, fs=FS)

    signal_plot_path = save_prototype_plot(proto_signal, class_idx, OUTPUT_DIR)
    psd_plot_path = save_psd_plot(proto_signal, FS, class_idx, OUTPUT_DIR)

    print(f"Class {class_idx} ({LABEL_NAMES.get(class_idx, class_idx)})")
    print(f"  Prototype class probabilities: {np.round(proto_probs, 4)}")
    print(f"  Prototype bandpower          : {bandpower_string(proto_bp)}")
    print(f"  Saved prototype signal plot  : {signal_plot_path}")
    print(f"  Saved prototype PSD plot     : {psd_plot_path}")
    print("-" * 60)

print("\nDone. Check the 'analysis_outputs' folder for saved plots.")