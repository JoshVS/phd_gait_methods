from classifiers.dg_stgcn import DGSTGCN, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# from copy_dataset_loaders.CASIA_dataset import CASIADataset
from dataset_loaders.RobberyDataset import RobberyDataset

import sys

def get_deep_size(obj, seen=None):
    """Recursively calculates the deep size of an object in bytes."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Object already counted

    seen.add(obj_id)
    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_deep_size(v, seen) for v in obj.values())
        size += sum(get_deep_size(k, seen) for k in obj.keys())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):  # Custom objects
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):  # Objects with __slots__
        for slot in obj.__slots__:
            try:
                size += get_deep_size(getattr(obj, slot), seen)
            except AttributeError:
                pass # Slot might not be set

    return size

SKIP_FRAMES = 5


def manual_batch(arr, batch_size=BATCH_SIZE):
    # print(arr.size())
    # quit()
    if type(arr) == list:
        return arr
    out_arr = [None] * (arr.size()[0] // batch_size)
    for i in range(arr.size()[0] // batch_size):
        out_arr[i] = arr[i * batch_size:(i + 1) * batch_size]
    if arr.size()[0] % batch_size != 0:
        out_arr.append(arr[(arr.size()[0] // batch_size) * batch_size:])



    return out_arr

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
    def __init__(self, ds, X, y, batch_size = BATCH_SIZE):
        self.ds = ds
        self.strat = ds.stratify_y
        self.batch_size = batch_size
        self.n_classes = ds.n_classes
        self.in_edge = ds.in_edge
        self.n_point = ds.X.shape[3]
        self.classes = ds.classes
        self.num_timesteps = ds.X.shape[2]
        self.in_channels = ds.X.shape[1]
        self.num_person = ds.X.shape[-1]
        self.X = manual_batch(X)
        self.y = manual_batch(y)
        self.num_batches = len(self.X)
        self.length = sum([x.size()[0] for x in self.X])
        # print(X.size()[2])
        # quit()

    
    def to(self, device):
        self.X = [x.to(device) for x in self.X]
        self.y = [y.to(device) for y in self.y]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

my_ds = RobberyDataset(generate_test_video=None, 
                    max_samples=None,
                    num_timesteps=30,
                    skip_frames=SKIP_FRAMES,
                    max_people=2,
                    max_samples_per_video=150)
# print(f"Dataset size: {get_deep_size(my_ds)/1e9} GB")
# quit()

VAL_SIZE = 0.3
X_train, X_val, y_train, y_val = train_test_split(my_ds.X, my_ds.y, test_size=VAL_SIZE, shuffle=True, stratify=my_ds.stratify_y)
train = TrainValDataset(my_ds, X_train, y_train)
val = TrainValDataset(my_ds, X_val, y_val)

ds = (train, val)

# ds = CASIATorchDataset(CASIADataset(generate_test_video=None, max_samples=10))
# print(dsiter.next())
classifier = DGSTGCN(ds)
