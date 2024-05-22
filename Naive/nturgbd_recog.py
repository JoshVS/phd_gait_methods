from dataset_loaders.NTURGBD_dataset import NTURGBDDataset
from classifiers.stgcn import STGCN
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch

class TrainValDataset(Dataset):
    def __init__(self, cache_dir="streaming", batch_size=32):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.file_index = 0
        self.X_cached = torch.tensor(np.load(os.path.join(cache_dir, "X_0.pkl")), dtype=np.double)
        self.y_cached = torch.tensor(np.load(os.path.join(cache_dir, "y_0.pkl")), dtype=np.double)
        # print(X.size()[2])
        # quit()

    
    def __len__(self):
        return len(os.listdir(self.cache_dir)) * self.batch_size
    
    def __getitem__(self, idx):
        if self.file_index != idx // self.batch_size:
            self.file_index = idx // self.batch_size
            X_filename = os.path.join(self.cache_dir, f"X_{self.file_index}.pkl")
            y_filename = os.path.join(self.cache_dir, f"y_{self.file_index}.pkl")
            self.X_cached = torch.tensor(np.load(X_filename), dtype=np.double)
            self.y_cached = torch.tensor(np.load(y_filename), dtype=np.double)

        return self.X_cached[idx % self.batch_size], self.y_cached[idx % self.batch_size]


my_ds = NTURGBDDataset(generate_test_video=None, 
                    max_samples=15,
                    max_classes=2, 
                    min_samples = 10, 
                    num_timesteps="pad")
print("SUCCESS")

# quit()
TEST_SIZE = 0.3
VAL_SIZE = 0.3



# X_train, X_test, y_train, y_test = train_test_split(my_ds.X, my_ds.y, test_size=TEST_SIZE, shuffle=True, stratify=my_ds.y)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, shuffle=True, stratify=y_train)
train = TrainValDataset(cache_dir=os.path.join("streaming", "train"))
test = TrainValDataset(cache_dir=os.path.join("streaming", "test"))
val = TrainValDataset(cache_dir=os.path.join("streaming", "val"))

ds = (train, test, val)

# ds = CASIATorchDataset(CASIADataset(generate_test_video=None, max_samples=10))
# print(dsiter.next())
classifier = STGCN(ds)
