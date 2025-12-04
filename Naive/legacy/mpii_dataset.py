from scipy.io import loadmat

class MPIIDataset:
    def __init__(self, directory="../../../../Datasets/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat"):

        X, y = self.get_file_data(directory)
        pass

    def get_file_data(self, directory):
        
        annots = loadmat(directory)
        print(annots["RELEASE"])
        print(annots.keys())
        return None, None



if __name__ == '__main__':
    ds = MPIIDataset()

    