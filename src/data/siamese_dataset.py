import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class SiameseFeatherDataset(Dataset):
    def __init__(self, feather_path):
        df = pd.read_feather(feather_path)

        # bỏ flow_id
        X = df.drop(columns=["flow_id", "Label"]).values
        y = df["Label"].values

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.label_index = {}
        for i, label in enumerate(self.y):
            self.label_index.setdefault(int(label), []).append(i)

        self.labels = list(self.label_index.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x1 = self.X[idx]
        y1 = self.y[idx]

        # positive / negative sampling
        if random.random() < 0.5:
            idx2 = random.choice(self.label_index[int(y1)])
            y = 1.0
        else:
            neg_label = random.choice([l for l in self.labels if l != int(y1)])
            idx2 = random.choice(self.label_index[neg_label])
            y = 0.0

        x2 = self.X[idx2]
        return x1, x2, torch.tensor(y, dtype=torch.float32)
