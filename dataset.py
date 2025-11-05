import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from config import cfg

class TabularCSVDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols=None, target_col=None):
        if feature_cols is None:
            feature_cols = [c.strip() for c in cfg.feature_cols]
        if target_col is None:
            target_col = cfg.target_col_final  

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        self.X = df[feature_cols].astype("float32").values
        self.y = df[target_col].astype("float32").values.reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.y[idx])
        return x, y
