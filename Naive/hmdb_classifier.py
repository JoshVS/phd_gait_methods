from dataset_loaders.HMDB_dataset import HMDBDataset
from classifiers.stgcn import STGCN
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
class CASIATorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.n_classes = ds.n_classes
        self.in_edge = ds.in_edge
        self.n_point = ds.X.shape[3]
        self.X = self.ds.X
        self.y = self.ds.y
        self.targets = self.ds.y

    def __len__(self):
        return self.ds.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TrainValDataset(Dataset):
    def __init__(self, ds, X, y):
        self.ds = ds
        self.n_classes = ds.n_classes
        self.in_edge = ds.in_edge
        self.n_point = ds.X.shape[3]
        self.X = X
        self.y = y
        # print(X.size()[2])
        # quit()

    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

my_ds = HMDBDataset(generate_test_video=None, max_samples=None)

X_train, X_test, y_train, y_test = train_test_split(my_ds.X, my_ds.y, test_size=0.1, shuffle=True, stratify=my_ds.y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, stratify=y_train)
train = TrainValDataset(my_ds, X_train, y_train)
test = TrainValDataset(my_ds, X_test, y_test)
val = TrainValDataset(my_ds, X_val, y_val)

ds = (train, test, val)

# ds = CASIATorchDataset(CASIADataset(generate_test_video=None, max_samples=10))
# print(dsiter.next())
classifier = STGCN(ds)
