import numpy as np
from dataset import NaiveKinectDataset
from naive_classifier import create_classifier, create_frame_level_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def manhattan_distance(a, b):
    d = 0
    for x, y in zip(a, b):
        d += abs(x - y)
    return d

class LinearAssignmentClassifier():
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = dataset.X_train
        self.y = dataset.y_train
        self.q = dataset.q_train

        self.n_features = dataset.X_train.shape[-1] // 3
        self.n_timesteps = dataset.X_train.shape[1]
        
        self.remove_one_hot()
        self.X, self.y, self.q = self.create_gallery(self.X, self.y, self.q)
        self.create_cost_matrix(dataset.X_test[0:8])


    def create_cost_matrix(self, X):
        cost_matrices = []
        loop = tqdm(range(X.shape[0]))
        print("Creating cost matrix")
        for i in loop:
            curr_cycle = i % 4
            curr_s = X[i]
            cost_matrix_for_this_test_sample = [] # Should be shape (n_timesteps, n_cols, 3)
            for t in range(curr_s.shape[0]):
                
                cost_matrix_for_this_timestamp = []
                for g in range(self.X[curr_cycle].shape[1]):
                        
                    curr_row = [
                        manhattan_distance(curr_s[t, 0::3], self.X[curr_cycle][:, g, 0]),
                        manhattan_distance(curr_s[t, 1::3], self.X[curr_cycle][:, g, 1]),                    
                        manhattan_distance(curr_s[t, 2::3], self.X[curr_cycle][:, g, 2])
                    ]
                    cost_matrix_for_this_timestamp.append(np.array(curr_row))
                cost_matrix_for_this_test_sample.append(np.array(cost_matrix_for_this_timestamp))
            cost_matrices.append(np.array(cost_matrix_for_this_test_sample))
        return cost_matrices



    def remove_one_hot(self):
        new_y = np.empty((self.y.shape[0],))
        for i in range(self.y.shape[0]):
            new_y[i] = np.argmax(self.y[i,:])
        self.y = new_y


    def create_gallery(self, X, y, q):
        x_vals = []
        y_vals = []
        q_vals = []
        for i in range(4):
            x_new, y_new, q_new = self._create_gallery(i, X, y, q)
            x_vals.append(x_new)
            y_vals.append(y_new)
            q_vals.append(q_new)
        return x_vals, y_vals, q_vals
    
    
    def _create_gallery(self, step, X, y, q, total_steps=4):
        # Feature Vector for Stage 1: (n_features//3, n_classes * n_samples * n_timesteps, 3)
        first_stages = X[step::total_steps, :, :]
        y_possibilities = y[step::total_steps]
        q_for_this_stage = q[step::total_steps,:]
        set_of_matrices = []
        set_of_qs = None
        y_selection = []
        for i in range(self.dataset.n_classes):
            
            curr_selection = first_stages[np.where(y_possibilities == i),...]
            curr_q = q_for_this_stage[np.where(y_possibilities == i),...]
            
            curr_selection = curr_selection.reshape(-1, X.shape[-1])
            curr_q = curr_q.reshape(-1)

            if set_of_qs is None:
                set_of_qs = curr_q
            else:
                set_of_qs = np.concatenate((set_of_qs, curr_q))

            for f in range(X.shape[-1] // 3):
                curr_feature_selection = curr_selection[:, 3 * f: 3 * f + 3]
                if len(set_of_matrices) <= f:
                    set_of_matrices.append(curr_feature_selection)
                else:
                    set_of_matrices[f] = np.concatenate((set_of_matrices[f], curr_feature_selection), axis=0)
            y_selection += [i] * curr_selection.shape[0]

        return np.array(set_of_matrices), y_selection, set_of_qs








ds = NaiveKinectDataset(max_samples=10)
classifier = LinearAssignmentClassifier(ds)
