from dataset import NaiveVideoDataset


skel_data = NaiveVideoDataset(max_samples=8)
print(skel_data.shape())