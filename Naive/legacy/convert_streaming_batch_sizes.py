import os
import numpy as np
from tqdm import tqdm

def convert_batch_sizes(old_size, new_size):
    def convert_set(set_name):
        if not os.path.exists(f"streaming{os.sep}{set_name}{os.sep}{new_size}"):
            os.makedirs(f"streaming{os.sep}{set_name}{os.sep}{new_size}")
        num_files = len(os.listdir(os.path.join("streaming", set_name, str(old_size)))) // 2
        new_file_idx = 0
        print(f"Converting {set_name}")
        loop = tqdm(range(num_files))
        for f in loop:
            curr_X_data = np.load(os.path.join("streaming", set_name, str(old_size), f"X_{f}.pkl.npy"))
            curr_y_data = np.load(os.path.join("streaming", set_name, str(old_size), f"y_{f}.pkl.npy"))
            for i in range(old_size // new_size):
                X_filename = os.path.join("streaming", set_name, str(new_size), f"X_{new_file_idx}.pkl")
                y_filename = os.path.join("streaming", set_name, str(new_size), f"y_{new_file_idx}.pkl")
                np.save(X_filename, curr_X_data[i * new_size: i * new_size + new_size])
                np.save(y_filename, curr_y_data[i * new_size: i * new_size + new_size])
                new_file_idx += 1
    
    convert_set("train")
    convert_set("test")
    convert_set("val")

convert_batch_sizes(1024, 32)

