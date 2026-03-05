import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torcheeg.models import EEGNet

from torcheeg.datasets import SleepEDFxDataset
from torcheeg import transforms
from torcheeg.models import EEGNet
from torch.utils.data import DataLoader

# 1) Dataset: cassette subset, 2 EEG channels, 30s windows
dataset = SleepEDFxDataset(
    root_path="./sleep-edf-database-expanded-1.0.0",
    version="cassette",
    num_channel=2,          # if this errors, try num_channels=2 (docs show both)
    chunk_size=3000,
    remove_unclear_example=True,
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
    transforms.Select(key="label"),  # <-- was "stage"
    transforms.Lambda(lambda s: 0 if ("W" in s or s == "W") else 1)
])
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2) Model: must match dataset sample shape (C, T) = (2, 3000)
model = EEGNet(
    chunk_size=3000,
    num_electrodes=2,
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=2
)

x, y = next(iter(loader))   # x: [B, 2, 3000]
print("x shape from loader:", x.shape)
x = x.unsqueeze(1)                      # -> [64, 1, 2, 3000]
print("x shape for EEGNet:", x.shape)
logits = model(x)
print("Input shape:", x.shape)
print("Output shape:", logits.shape)
'''
class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    '''