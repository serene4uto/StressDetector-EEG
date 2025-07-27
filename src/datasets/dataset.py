import os
import pickle
from torch.utils.data import Dataset

# Load the dataset to RAM in once

class ZekiScalarDataset(Dataset):
    def __init__(self, dataset_path, transform=None, target_transform=None):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        self.info = "Zeki Set"
            
        self.labels = dataset['labels']
        self.data = dataset['dataset']

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data.tolist())
    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        label_item = self.labels[idx]
        if self.transform:
            data_item = self.transform(data_item)
        if self.target_transform:
            label_item = self.target_transform(label_item)
        return data_item, label_item
