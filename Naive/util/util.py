import torch
import os
import numpy as np
class BatchedDataset(torch.Tensor):
    def __init__(self, dataset_path, batch_size, total_size, read_path):
        self.dataset_path = dataset_path
        self.read_path = read_path
        self.total_size = total_size
        self.batch_size = batch_size
        self.num_batches = (os.listdir(dataset_path))
        self.indices = self.shuffle()

        super(BatchedDataset, self).__init__()

    def shuffle(self):
        return np.random.shuffle(list(range(self.total_size)))
    
    def batch_indices(self):
        curr_batch = []
        for i in range(self.total_size):
            with open(os.path.join(self.read_path, f"data_{i}.pkl"), 'rb') as f:
                data = torch.load(f)
                curr_batch.append(data)
        
    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset_path))
        batch = [self.dataset_path[i] for i in range(start_idx, end_idx)]
        return torch.utils.data.dataloader.default_collate(batch)