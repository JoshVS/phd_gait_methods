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
        
        self.y = self.remove_one_hot(self.y)
        self.X, self.y, self.q = self.create_gallery(self.X, self.y, self.q) # (n_features//3, n_classes * n_samples * n_timesteps, 3)
        cm = self.create_cost_matrix(dataset.X_test[0:8])
        y_test_new = self.remove_one_hot(dataset.y_test)
        

        y_pred = self.classify_gait(cm)
        print(y_test_new[0:8])
        print(self.classify_individual_steps(y_pred))
        quit()

    def classify_individual_steps(self, gait_classifications):
        ret_vals = []
        for i in range(len(gait_classifications)):
            values, counts = np.unique(gait_classifications[i], return_counts=True)
            ret_vals.append(values[counts.argmax()])
        return ret_vals



    def classify_gait(self, cm):
        ret_vals = []
        loop = tqdm(range(len(cm)))
        for i in loop:
            ret_vals.append(self.hungarian_algorithm(cm, self.y, i))
        return ret_vals


    def hungarian_algorithm(self, cost_matrix, labels, sample_number):
        gait_phase = sample_number % 4
        # Takes in cost matrix array of shape (n_samples, n_columns, 3)
        # Multiplies with quality matrix
        # Returns Y, which should be individual votes
        # Y should only be shape (3,)
        overall_solutions = []
        for t in range(cost_matrix[sample_number].shape[0]): # Per timestep
            x_arr = cost_matrix[sample_number][t,:,0]
            y_arr = cost_matrix[sample_number][t,:,1]
            z_arr = cost_matrix[sample_number][t,:,2]

            cost_grid = np.meshgrid(x_arr, y_arr, z_arr, copy=True)
            for g in range(len(cost_grid)):
                for i in range(cost_matrix[sample_number].shape[1]):
                    cost_grid[g][i,i,i] = np.inf
                cost_grid[g] = cost_grid[g]
            sum_grids = None
            for g in cost_grid:
                if sum_grids is None:
                    sum_grids = g
                else:
                    sum_grids = sum_grids + g
            x_choices, y_choices, z_choices = np.where(sum_grids == np.min(sum_grids))
            x_choice, y_choice, z_choice = x_choices[0], y_choices[0], z_choices[0]
            values, counts = np.unique([labels[gait_phase][x_choice], labels[gait_phase][y_choice], labels[gait_phase][z_choice]], return_counts=True)
            overall_solutions.append(values[counts.argmax()])
        return overall_solutions



    def create_cost_matrix(self, X):
        cost_matrices = []
        loop = tqdm(range(X.shape[0]))
        print("Creating cost matrix")
        for i in loop:
            curr_cycle = i % 4
            curr_s = X[i]
            cost_matrix_for_this_test_sample = [] # Should be shape (n_timesteps, n_cols, 3)
            for t in range(curr_s.shape[0]): # Per timestep
                
                cost_matrix_for_this_timestamp = []
                for g in range(self.X[curr_cycle].shape[1]): # Per column
                    curr_row = [
                        manhattan_distance(curr_s[t, 0::3], self.X[curr_cycle][:, g, 0]),
                        manhattan_distance(curr_s[t, 1::3], self.X[curr_cycle][:, g, 1]),                    
                        manhattan_distance(curr_s[t, 2::3], self.X[curr_cycle][:, g, 2])
                    ]
                    cost_matrix_for_this_timestamp.append(np.array(curr_row))
                cost_matrix_for_this_test_sample.append(np.array(cost_matrix_for_this_timestamp))

            cost_matrices.append(np.array(cost_matrix_for_this_test_sample))
        return cost_matrices



    def remove_one_hot(self, y):
        new_y = np.empty((y.shape[0],))
        for i in range(y.shape[0]):
            new_y[i] = np.argmax(y[i,:])
        return new_y


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








ds = NaiveKinectDataset(max_samples=3)
classifier = LinearAssignmentClassifier(ds)
