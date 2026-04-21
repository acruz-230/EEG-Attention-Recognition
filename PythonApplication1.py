import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torcheeg.models import EEGNet
import sys
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import confusion_matrix, classification_report


log_file = open("F3training_output.txt", "w")
sys.stdout = log_file


# Settings
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cpu")
print("Using device:", device)

FS = 100
OUTPUT_DIR = "analysis_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_NAMES = {
    0: "NoMusic_NoQuestion",
    1: "Music_NoQuestion",
    2: "NoMusic_Question",
    3: "Music_Question"
}


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

        if X.ndim != 2:
            raise ValueError(
                f"Expected X to be 2D [epochs, time], but got shape {X.shape}"
            )

      
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

        # Normalize each epoch
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
  



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
class_counts = np.bincount(dataset.Y, minlength=num_classes)
print("Class counts:", class_counts)

class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
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

def compute_bandpower(epoch, fs=100):
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

def predict_single_epoch(model, x, device):
    model.eval()
    with torch.no_grad():
        x_model = x.unsqueeze(0).unsqueeze(1).to(device).float()   # [1,1,1,T]
        logits = model(x_model)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs

def compute_saliency(model, x, device, target_class=None):
    model.eval()

    x_in = x.unsqueeze(0).unsqueeze(1).to(device).float()   # [1,1,1,T]
    x_in.requires_grad_(True)

    logits = model(x_in)
    pred_class = int(torch.argmax(logits, dim=1).item())

    used_class = pred_class if target_class is None else int(target_class)

    score = logits[0, used_class]
    model.zero_grad()
    score.backward()

    saliency = x_in.grad.detach().abs().cpu().numpy().squeeze()
    logits_np = logits.detach().cpu().numpy().squeeze()

    return saliency, pred_class, used_class, logits_np

def top_saliency_region(saliency, window=200):
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
    plt.plot(
        t,
        saliency / (np.max(saliency) + 1e-8) * np.std(epoch_signal),
        label="Normalized saliency"
    )
    plt.title(f"Sample {sample_idx} | Pred={pred_class} | Actual={actual_class}")
    plt.xlabel("Time samples")
    plt.ylabel("Amplitude / relative saliency")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"saliency_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

def total_variation_loss(x):
    return torch.mean(torch.abs(x[..., 1:] - x[..., :-1]))

def optimize_class_prototype(
    model,
    target_class,
    signal_len,
    device,
    steps=300,
    lr=0.01,
    l2_weight=1e-4,
    tv_weight=1e-4
):
    model.eval()

    x_opt = torch.randn((1, 1, 1, signal_len), device=device, requires_grad=True)
    opt = optim.Adam([x_opt], lr=lr)

    for _ in range(steps):
        opt.zero_grad()

        logits = model(x_opt)
        class_score = logits[0, target_class]

        l2 = torch.mean(x_opt ** 2)
        tv = total_variation_loss(x_opt)

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

#Print random samples with guess and results from trained model
print("\n20 random predictions vs actual labels + bandpower:\n")

rand_samp = min(20, len(dataset))
indices = random.sample(range(len(dataset)), rand_samp)

print("\n===== SALIENCY / MODEL INFLUENCE =====\n")

saliency_indices = random.sample(range(len(dataset)), min(5, len(dataset)))

for idx in saliency_indices:
    x, y = dataset[idx]
    epoch_np = x.squeeze().numpy()

    saliency, pred_class, used_class, logits_np = compute_saliency(model, x, device)
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
    print(f"  Actual class           : {y.item()} ({LABEL_NAMES.get(int(y.item()), y.item())})")
    print(f"  Predicted class        : {pred_class} ({LABEL_NAMES.get(pred_class, pred_class)})")
    print(f"  Used class for grad    : {used_class}")
    print(f"  Logits                 : {np.round(logits_np, 4)}")
    print(f"  Most influential region: samples {start} to {end}")
    print(f"  Influential bandpower  : {bandpower_string(influential_bp)}")
    print(f"  Saved saliency plot    : {plot_path}")
    print("-" * 60)

    print("\n===== CLASS PROTOTYPE SIGNALS =====\n")

signal_len = dataset.X.shape[1]

for class_idx in range(num_classes):
    proto_signal, proto_probs = optimize_class_prototype(
        model=model,
        target_class=class_idx,
        signal_len=signal_len,
        device=device,
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

with torch.no_grad():
    for i in indices:
        x_eeg, y_labels = dataset[i]
        epoch_np = x_eeg.squeeze().numpy()

        bp = compute_bandpower(epoch_np, fs=FS)
        guess, probs = predict_single_epoch(model, x_eeg, device)

        actual_name = LABEL_NAMES.get(int(y_labels.item()), str(int(y_labels.item())))
        pred_name = LABEL_NAMES.get(int(guess), str(int(guess)))

        print(f"Sample {i}")
        print(f"  Actual       : {y_labels.item()} ({actual_name})")
        print(f"  Predicted    : {guess} ({pred_name})")
        print(f"  Confidence   : {probs[guess]:.4f}")
        print(f"  Probabilities: {np.round(probs, 4)}")
        print(f"  Bandpower    : {bandpower_string(bp)}")
        print("-" * 60)

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




log_file.close()