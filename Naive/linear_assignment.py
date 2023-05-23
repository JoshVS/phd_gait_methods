import numpy as np
from dataset import NaiveKinectDataset
from naive_classifier import create_classifier, create_frame_level_classifier
import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from scipy.optimize import linear_sum_assignment as hungarian_algorithm_method
# from hungarian import hungarian_algorithm_method

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
        cm = self.create_cost_matrix(dataset.X_test)
        y_test_new = self.remove_one_hot(dataset.y_test)
        gait_predictions = self.classify_gait(cm, hungarian=True)
        
        # gait_predictions = self.classify_gait(cm)
        # if gait_validation != gait_predictions:
        #     print("Non-Matching Solutions. Hungarian Algorithm first, then Brute")
        #     print(gait_validation)
        #     print(gait_predictions)
        #     quit()
        # assert gait_predictions == gait_validation
        y_pred = self.classify_individual_steps(gait_predictions)
        print(y_test_new)
        print(y_pred)
        self.report_metrics(y_test_new, y_pred)

        quit()

    def report_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        accuracy = accuracy_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)

        out_str = f"""###############################################
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1: {f1:.3f}
        Accuracy: {accuracy:.3f}

###############################################
        """
        with open("results.txt", 'w') as outfile:
            outfile.write(out_str)
        print(out_str)

        sns.heatmap(cm, annot=False)
        plt.savefig("confusion_matrix.png")




    def classify_individual_steps(self, gait_classifications):
        ret_vals = []
        for i in range(len(gait_classifications)):
            values, counts = np.unique(gait_classifications[i], return_counts=True)
            ret_vals.append(values[counts.argmax()])
        return ret_vals



    def classify_gait(self, cm, hungarian=False):
        ret_vals = []
        loop = tqdm(range(len(cm)))
        for i in loop:
            if hungarian:
                ret_vals.append(self.hungarian_algorithm(cm, self.y, i))
            else:
                ret_vals.append(self.brute_algorithm(cm, self.y, i))    
        return ret_vals

    def hungarian_algorithm(self, cost_matrix, labels, sample_number):
        gait_phase = sample_number % 4
        # Takes in cost matrix array of shape (n_samples, n_columns, 3)
        # Multiplies with quality matrix
        # Returns Y, which should be individual votes
        # Y should only be shape (3,)
        overall_solutions = []
        # print(cost_matrix[sample_number].shape)
        # print(len(cost_matrix))
        # quit()
        for t in range(cost_matrix[sample_number].shape[0]): # Per timestep


            columns, rows = hungarian_algorithm_method(cost_matrix[sample_number][t,:,:])
            # print(pos)
            # quit()
            choices = [None, None, None]
            for c, r in zip(columns, rows):
                choices[r] = c
            x_choice, y_choice, z_choice = choices[0], choices[1], choices[2]
            values, counts = np.unique([labels[gait_phase][x_choice], labels[gait_phase][y_choice], labels[gait_phase][z_choice]], return_counts=True)
            overall_solutions.append(values[counts.argmax()])
        return overall_solutions


    def brute_algorithm(self, cost_matrix, labels, sample_number):
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
        # print(X.shape)
        # quit()
        cost_matrices = []
        loop = tqdm(range(X.shape[0]))
        print("Creating cost matrix")
        for i in loop:
            curr_cycle = i % 4
            curr_s = X[i]
            cost_matrix_for_this_test_sample = [] # Should be shape (n_timesteps, n_cols, 3)
            
            cost_matrix_for_this_test_sample_test = [] # Should be shape (n_timesteps, n_cols, 3)
            to_add = curr_s.reshape(-1, curr_s.shape[-1]//3, 3)

            feature_vector = np.transpose(self.X[curr_cycle], (1,0, 2)).astype(np.float64)
            
            for t in range(to_add.shape[0]): # Per timestep
                # print(to_add.dtype, feature_vector.dtype, self.X[curr_cycle].dtype)
                # quit()
                # cost_matrix_for_this_timestamp_test = np.abs(to_add[t,...].astype(np.float64) - feature_vector).sum(axis=1) #self.X[curr_cycle].reshape((self.X[curr_cycle].shape[1],) + to_add[t,...].shape))
                # test = None
                # for i1 in range(to_add.shape[1]):
                #     for i2 in range(to_add.shape[2]):
                #         if test is None:
                #             test = np.abs(self.X[curr_cycle][i1, :, :] - to_add[t, i1, :]).astype(np.float64)
                #         else:
                #             test += np.abs(self.X[curr_cycle][i1, :, :] - to_add[t, i1, :]).astype(np.float64)

                # print(test == cost_matrix_for_this_timestamp_test)
                # print(test.dtype, cost_matrix_for_this_timestamp_test.dtype)
                # quit()
                # cost_matrix_for_this_timestamp_test = test

                # print(cost_matrix_for_this_timestamp_test.shape)
                # quit()
                # cost_matrix_for_this_timestamp_test = cost_matrix_for_this_timestamp_test.sum(axis=1)
                # print(cost_matrix_for_this_timestamp_test.shape)
                # quit()
                # cost_matrix_for_this_timestamp = subtractions_per_feature
                # print(cost_matrix_for_this_timestamp_test.shape, to_add.shape)
                # quit()
                # a = None
                # for f in range(cost_matrix_for_this_timestamp_test.shape[1]):
                #     if a is None:
                #         a = cost_matrix_for_this_timestamp_test[:,f,:]
                #     else:
                #         a = a + cost_matrix_for_this_timestamp_test[:,f,:]
                # cost_matrix_for_this_timestamp_test = a

                
                cost_matrix_for_this_timestamp = []
                # print(self.X[curr_cycle].shape)
                # print(curr_s.shape)
                # quit()
                cost_matrix_for_this_timestamp = np.array([np.abs(curr_s[t, x::3] - self.X[curr_cycle][:, :, x].T).astype(np.float64).sum(axis=1) for x in range(3)]).T
                # test2 = 

                # for g in range(self.X[curr_cycle].shape[1]): # Per column
                #     curr_row = [
                #         manhattan_distance(curr_s[t, 0::3], self.X[curr_cycle][:, g, 0]),
                #         manhattan_distance(curr_s[t, 1::3], self.X[curr_cycle][:, g, 1]),
                #         manhattan_distance(curr_s[t, 2::3], self.X[curr_cycle][:, g, 2])
                #     ]
                #     x_test = manhattan_distance(curr_s[t, 0::3], self.X[curr_cycle][:, g, 0])
                #     curr_row = [np.abs(curr_s[t, x::3] - self.X[curr_cycle][:, g, x]).astype(np.float64).sum() for x in range(3)]
                #     # curr_row = np.abs(curr_s.reshape(-1, 8, 3)[t,...] - self.X[curr_cycle][:, g, :])
                #     # print(curr_row)
                #     # quit()
                #     cost_matrix_for_this_timestamp.append(np.array(curr_row).astype(np.float64))
                # print((np.array(cost_matrix_for_this_timestamp) == cost_matrix_for_this_timestamp_test))
                # print(np.array(cost_matrix_for_this_timestamp).dtype, cost_matrix_for_this_timestamp_test.dtype)
                # print(test2.shape)
                # print((np.array(cost_matrix_for_this_timestamp) == test2).all() )
                # print(cost_matrix_for_this_timestamp_test.shape, np.array(cost_matrix_for_this_timestamp).shape)
                
                # print((np.array(cost_matrix_for_this_timestamp) - cost_matrix_for_this_timestamp_test))
                # print(np.where(np.array(cost_matrix_for_this_timestamp) == cost_matrix_for_this_timestamp_test))
                # quit()

                cost_matrix_for_this_test_sample.append(np.array(cost_matrix_for_this_timestamp))
                # cost_matrix_for_this_test_sample.append(cost_matrix_for_this_timestamp_test)
                # cost_matrix_for_this_test_sample_test.append(cost_matrix_for_this_timestamp_test)
                # print(cost_matrix_for_this_timestamp_test.shape)
                # quit()

            # print(np.array(cost_matrix_for_this_test_sample_test).shape)
            # print(np.array(cost_matrix_for_this_test_sample).shape)
            # print(np.array(cost_matrix_for_this_test_sample) == np.array(cost_matrix_for_this_test_sample_test))
            # quit()
            # cost_matrices.append(np.array(cost_matrix_for_this_test_sample))
            cost_matrices.append(np.array(cost_matrix_for_this_test_sample))
        # print()
        # quit()
        return cost_matrices



    def remove_one_hot(self, y):
        new_y = np.empty((y.shape[0],))
        for i in range(y.shape[0]):
            new_y[i] = np.argmax(y[i,:])
        return new_y.astype(np.int32)


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








ds = NaiveKinectDataset(max_samples=40)
# print(ds.n_classes)
# quit()
classifier = LinearAssignmentClassifier(ds)
