import torch
from torch.utils.data import Dataset


class Feeder(Dataset):
    """Feeder for single inputs"""

    def __init__(self, data, label, transforms=None):
        self.data = data
        self.label = label
        self.transforms = transforms

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data = self.data[index]  # NCTV
        label = self.label[index]

        # data augumentation
        if self.transforms:
            data = self.transforms(data)

        # numpy to tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        return data, label
