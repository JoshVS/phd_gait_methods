from dataset_loaders.HMDB_dataset import HMDBDataset
from classifiers.stgcn import STGCN, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
        self.length = sum([x.size()[0] for x in self.X])
        # print(X.size()[2])
        # quit()

    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

exclude = [
    'brush_hair',
    'chew',
    'catch',
    'eat',
    'smoke',
    'laugh',
    'kiss',
    'hug',
    'cartwheel',
    'draw_sword',
    'fall_floor',
    'run',
    'turn',
    'sit', 
    'dive',
    'jump',
    "climb",
    'climb_stairs',
    'hit',
    'kick',
    'kick_ball',
    'pullup',
    'punch',
    'push',
    'ride_bike',
    'ride_horse',
    'shake_hands',
    'shoot_gun'

]
exclude=None
my_ds = HMDBDataset(generate_test_video=None, 
                    max_samples=None,
                    max_classes=None, 
                    min_samples = 10, 
                    exclude_classes=exclude, 
                    num_timesteps=10)

TEST_SIZE = 0.3
VAL_SIZE = 0.3

X_train, X_test, y_train, y_test = train_test_split(my_ds.X, my_ds.y, test_size=TEST_SIZE, shuffle=True, stratify=my_ds.y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, shuffle=True, stratify=y_train)
train = TrainValDataset(my_ds, X_train, y_train)
test = TrainValDataset(my_ds, X_test, y_test)
val = TrainValDataset(my_ds, X_val, y_val)

ds = (train, test, val)

# ds = CASIATorchDataset(CASIADataset(generate_test_video=None, max_samples=10))
# print(dsiter.next())
classifier = STGCN(ds)
