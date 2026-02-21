import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torcheeg.models import EEGNet

class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

data = scipy.io.loadmat('eeg_data.mat')

X = data['X']  
Y = data['Y']

dataset = EEGDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = EEGNet(
    chunk_size=128,      
    num_electrodes=14,   
    dropout=0.5,
    kernel_1=64,
    kernel_2=16,
    F1=8,
    F2=16,
    D=2,
    num_classes=2
)

x, y = next(iter(loader))
output = model(x)
print(output.shape)

class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long).squeeze()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]