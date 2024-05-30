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
        self.X_cached = torch.tensor(np.load(os.path.join(cache_dir, str(batch_size), "X_0.pkl.npy")), dtype=torch.double)
        self.y_cached = torch.tensor(np.load(os.path.join(cache_dir, str(batch_size), "y_0.pkl.npy")), dtype=torch.double)
        self.num_timesteps = self.X_cached.shape[2]
        self.n_classes = self.y_cached.shape[1]
        self.classes = [f"class_{x}" for x in range(self.n_classes)]
        self.n_point = self.X_cached.shape[3]
        keypoints_arr = [
            "center_hip",
            "torso",
            "neck",
            "head",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hand",

            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "right_hand",

            "left_hip",
            "left_knee",
            "left_ankle",
            "left_foot",

            "right_hip",
            "right_knee",
            "right_ankle",
            "right_foot",

            "center_chest",

            "left_index",
            "left_thumb",

            
            "right_index",
            "right_thumb",

        ]
        self.kp_indices = {}
        for i, s in enumerate(keypoints_arr):
            self.kp_indices[s] = i
        self.connections = [
            ('center_hip', 'torso'),
            ('center_hip', 'left_hip'),
            ('center_hip', 'right_hip'),


            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_foot'),

            
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_foot'),
            

            ('torso', 'center_chest'),
            ('center_chest', 'neck'),
            ('neck', 'head'),
            ('center_chest', 'left_shoulder'),
            ('center_chest', 'right_shoulder'),

            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_hand'),
            ('left_hand', 'left_index'),
            ('left_hand', 'left_thumb'),

            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_hand'),
            ('right_hand', 'right_index'),
            ('right_hand', 'right_thumb')
        ]

        self.edge_matrix = [[self.kp_indices[x] for x, _ in self.connections], [self.kp_indices[y] for _, y in self.connections]]
        self.in_edge = [
            (self.kp_indices[x], self.kp_indices[y])
            for (x, y) in self.connections
        ]        

        self.upper_torso = [
            "head",
            "neck",
            "center_chest",
            
            "left_shoulder",
            "right_shoulder",
        ]

        self.lower_torso = [
            "left_hip",
            "center_hip",
            "right_hip"
        ]
        self.headpoint = "center_hip"
        self.left_elbow = "left_elbow"
        self.right_elbow = "right_elbow"
        self.left_knee = "left_knee"
        self.right_knee = "right_knee"
        self.right_wrist = "right_wrist"
        self.left_wrist = "left_wrist"
        self.left_ankle = "left_ankle"
        self.right_ankle = "right_ankle"


    
    def __len__(self):
        return len(os.listdir(self.cache_dir)) * self.batch_size // 2
    
    def __getitem__(self, idx):
        if self.file_index != idx // self.batch_size:
            # print("Getting new batch")
            self.file_index = idx // self.batch_size
            X_filename = os.path.join(self.cache_dir,str(self.batch_size),  f"X_{self.file_index}.pkl.npy")
            y_filename = os.path.join(self.cache_dir,str(self.batch_size),  f"y_{self.file_index}.pkl.npy")
            self.X_cached = torch.tensor(np.load(X_filename), dtype=torch.double)
            self.y_cached = torch.tensor(np.load(y_filename), dtype=torch.double)

        return self.X_cached[idx % self.batch_size], self.y_cached[idx % self.batch_size]


# my_ds = NTURGBDDataset(generate_test_video=None, 
#                     max_samples=None,
#                     max_classes=None, 
#                     min_samples = 10, 
#                     num_timesteps="pad",
#                     save_batch_size=32)
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
