from dataset_loaders.casia_dataset import CASIADataset
from classifiers.stgcn import STGCN
from torch.utils.data import Dataset, DataLoader

class CASIATorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.n_classes = ds.n_classes
        self.in_edge = ds.in_edge
        self.n_point = ds.X.shape[3]
        self.X = self.ds.X
        self.y = self.ds.y

    def __len__(self):
        return self.ds.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


ds = CASIATorchDataset(CASIADataset(generate_test_video=None, max_samples=0.5))
# print(dsiter.next())
classifier = STGCN(ds)
