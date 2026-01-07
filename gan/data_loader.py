import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class WarehouseItemDataset(Dataset):
    def __init__(self, csv_file):
        # Load data using Pandas
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: File {csv_file} not found.")
            df = pd.DataFrame(columns=['length', 'width', 'height', 'weight'])

        # Features: length, width, height, weight
        # Ensure extraction deals with potential missing values or bad types
        features = df[['length', 'width', 'height', 'weight']].apply(pd.to_numeric, errors='coerce').dropna()
        
        # Filter positive values
        features = features[
            (features['length'] > 0) & 
            (features['width'] > 0) & 
            (features['height'] > 0) & 
            (features['weight'] > 0)
        ]

        if features.empty:
            print("Warning: No valid items found in dataset!")
            # Dummy data to prevent crash
            self.data = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        else:
            self.data = features.values.astype(np.float32)

        # Normalize data to [0, 1] range
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
    
    def get_scaler(self):
        return self.scaler

def get_dataloader(csv_file, batch_size=32):
    dataset = WarehouseItemDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.get_scaler()
