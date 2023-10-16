from torch.utils.data import Dataset, DataLoader
import torch


class CodeDataset(Dataset):
    def __init__(self, data, label, language):
        self.data = data
        self.label = label
        self.language = language

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.language[self.label[index][0]])
