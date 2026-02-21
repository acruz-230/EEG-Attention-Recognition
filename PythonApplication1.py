from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

dataset = DEAPDataset(root_path='./data_preprocessed_python',
                      online_transform=transforms.Compose([
                          transforms.To2d(),
                          transforms.ToTensor()
                      ]),
                      label_transform=transforms.Compose([
                          transforms.Select(['valence', 'arousal']),
                          transforms.Binary(5.0),
                          transforms.BinariesToCategory()
                      ]))
print(dataset[0])
# EEG signal (torch.Tensor[1, 32, 128]),
# coresponding baseline signal (torch.Tensor[1, 32, 128]),
# label (int)