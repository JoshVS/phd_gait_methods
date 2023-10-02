import numpy as np
from dataset_loaders.genericdataset import dim
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
    def __init__(self, dataset, num_dims=3, num_phases=4):
        self.num_phases = num_phases
        self.num_dims = num_dims
        self.dataset = dataset
        self.X = dataset.X_train
        
        self.y = dataset.y_train
        self.q = dataset.q_train

        self.n_features = dim(dataset.X_train)[-1] // num_dims
        self.n_timesteps = dim(dataset.X_train)[1]
        
        # self.y = self.remove_one_hot(self.y)
        self.X, self.y, self.q = self.create_gallery(self.X, self.y, self.q) # (n_features//3, n_classes * n_samples * n_timesteps, 3)
        
        
    def generate_test_set_results(self):
        y_test_new = self.dataset.y_test #self.remove_one_hot(self.dataset.y_test)
        y_pred = self.predict(self.dataset.X_test, self.dataset.q_test)
        self.report_metrics(y_test_new, y_pred)

    def predict(self, X, q, use_hungarian=True):
        cm = self.create_cost_matrix(X)
        cm = self.adjust_cost_for_quality(cm)
        return self.classify_gait(cm, q, hungarian=use_hungarian)


    def balance_classes(self, X, y, q):
        for i in range(len(X)):
            X[i], y[i], q[i] = self._balance_classes(X[i], y[i], q[i])
        return X, y, q

    def _balance_classes(self, X, y, q):
        classes, class_counts = np.unique(y, return_counts=True)
        c_limiter = np.min(class_counts)
        limits = []
        for c, count in zip(classes, class_counts):
            curr_limiter = c_limiter * 2 if count > c_limiter * 2 else count
            limits.append(curr_limiter)
            start_ind = sum(limits) + curr_limiter
            end_ind = (sum(limits)) + count
            X = np.delete(X, list(range(start_ind, end_ind)), axis=1)            
            q = np.delete(q, list(range(start_ind, end_ind)), axis=0)
            y = np.delete(y, list(range(start_ind, end_ind)), axis=0)
        return X, y, q
        # print(X.shape)
        # print(len(classes) * c_limiter)
        # quit()


    def adjust_cost_for_quality(self, X):
        for i in range(len(X)):
            for j in range(X[i].shape[0]):
                for k in range(X[i].shape[2]):
                    X[i][j,:, k] /= self.q[i%self.num_phases]
        # for i in range(len(X)):
        #     for j in range(X[i].shape[0]):
        #         for k in range(X[i].shape[2]):
        #             X[i][j,:,k] /= self.q[i]
        return X


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

        sns.heatmap(cm, annot=(cm.shape[0] <= 8), xticklabels=self.dataset.classes, yticklabels=self.dataset.classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig("confusion_matrix.png")
        plt.close()


        n_classes = len(np.unique(y_true))
        # print(type(y_true))
        # quit()
        true_counts = [len(y_true[np.where(y_true == x)]) for x in np.unique(y_true)]
        trues_and_falses = [a == b for (a, b) in zip(y_true, y_pred)]
        predicted_counts = []
        for i in range(len(np.unique(y_true))):
            tmp = []
            for j in range(len(np.where(y_true == np.unique(y_true)[i])[0])):
                tmp.append(trues_and_falses[j])
            predicted_counts.append(sum(tmp))

        # predicted_counts = [sum(trues_and_falses[np.where(y_true == x)]) for x in np.unique(y_true)]
        out_data = [a / b for (a, b) in zip(predicted_counts, true_counts)]
        plt.figure()
        sns.barplot(x=np.arange(n_classes),y= out_data)
        plt.savefig("barplot.png")
        plt.close()




    def classify_individual_steps(self, gait_classifications):
        ret_vals = []
        for i in range(len(gait_classifications)):
            values, counts = np.unique(gait_classifications[i], return_counts=True)
            ret_vals.append(values[counts.argmax()])
        return ret_vals



    def classify_gait(self, cm, q, hungarian=False):
        ret_vals = []
        vote_vals = []
        print("Running Linear Matching Algorithm")
        loop = tqdm(range(len(cm)))
        for i in loop:
            if hungarian:
                vote_vals.append(self.hungarian_algorithm(cm, self.y, i, q))
                if len(vote_vals) == self.num_phases:
                    values, counts = np.unique(vote_vals, return_counts=True)
                    # ret_vals.extend([values[counts.argmax()] ] * self.num_phases)
                    ret_vals.extend(vote_vals)
                    vote_vals = []
            else:
                vote_vals.append(self.brute_algorithm(cm, self.y, i))
        values, counts = np.unique(vote_vals, return_counts=True)
        # ret_vals.extend([values[counts.argmax()]] * len(vote_vals))
        ret_vals.extend(vote_vals)
        return ret_vals

    def hungarian_algorithm(self, cost_matrix, labels, sample_number, q):
        gait_phase = sample_number % self.num_phases
        # Takes in cost matrix array of shape (n_samples, n_columns, 3)
        # Multiplies with quality matrix
        # Returns Y, which should be individual votes
        # Y should only be shape (3,)
        overall_solutions = []
        # print(cost_matrix[sample_number].shape)
        # print(len(cost_matrix))
        # quit()
        matching_scores = []
        s_acc = [0] * len(np.unique(labels[gait_phase]))
        for t in range(cost_matrix[sample_number].shape[0]): # Per timestep


            columns, rows = hungarian_algorithm_method(cost_matrix[sample_number][t,:,:])
            # print(pos)
            # quit()
            choices = [None] * self.num_dims
            for c, r in zip(columns, rows):
                choices[r] = c
            # print([labels[gait_phase][x_choice], labels[gait_phase][y_choice], labels[gait_phase][z_choice]])
            # print(np.where(np.array(labels[gait_phase]) == labels[gait_phase][x_choice]))
            # print(cost_matrix[sample_number][t,...].shape)
            # quit()
            change_indices = [np.where(np.array(labels[gait_phase]) == labels[gait_phase][a])[0] for a in choices]

            first_score = sum([cost_matrix[sample_number][t, choices[x], x] for x in range(len(choices))])
            for a, b in enumerate(choices):
                cost_matrix[sample_number][t,b,a] = np.inf

            
            columns, rows = hungarian_algorithm_method(cost_matrix[sample_number][t,:,:])
            
            second_choices = [None] * self.num_dims
            for c, r in zip(columns, rows):
                second_choices[r] = c

            # second_x_choice, second_y_choice, second_z_choice = second_choices[0], second_choices[1], second_choices[2]


            second_score = sum([cost_matrix[sample_number][t, second_choices[x], x] for x in range(len(second_choices))])
            s_similarity = 1 / first_score
            s_margin = 0 if first_score >= second_score else second_score / first_score

            

            s_total = q[sample_number][t] * s_similarity * s_margin

            for a in choices:
                # print(a)
                # print(s_total, s_similarity, s_margin, first_score)
                # print(labels[gait_phase][a], len(s_acc))
                s_acc[labels[gait_phase][a]] += s_total 
            
            matching_scores.append(s_total)

            values, counts = np.unique([labels[gait_phase][a] for a in choices], return_counts=True)
            overall_solutions.append(values[counts.argmax()])
            # TODO: s_similarity: 1 / column
            # TODO: Second Choice: Remove all choices for current class
            # TODO: S_margin: second_choice / first_choice if first_choice < second_choice, 0 otherwise
            # TODO: matching_score[t] = quality * s_similarity * s_margin
        # TODO: max(matching_score)
        # values, counts = np.unique(s_acc, return_counts=True)
        # print(s_acc)
        # quit()
        return np.argmax(s_acc)
        # return #overall_solutions[np.argmax(matching_scores)]


    def brute_algorithm(self, cost_matrix, labels, sample_number):
        gait_phase = sample_number % self.num_phases
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
            # print(np.where(labels[gait_phase] == labels[gait_phase][x_choice]))
            # quit()
            values, counts = np.unique([labels[gait_phase][x_choice], labels[gait_phase][y_choice], labels[gait_phase][z_choice]], return_counts=True)
            overall_solutions.append(values[counts.argmax()])
        return overall_solutions



    def create_cost_matrix(self, X):
        cost_matrices = []
        print("Creating cost matrix")
        loop = tqdm(range(len(X)))
        for i in loop:
            curr_cycle = i % self.num_phases
            curr_s = X[i]
            cost_matrix_for_this_test_sample = [] # Should be shape (n_timesteps, n_cols, 3)

            # print(dim(curr_s))
            # quit()
            # tmp = []
            # for a in curr_s:
            #     tmp.append([
            #         a[x::dim(curr_s)[-1] // self.num_dims]
            #         for x in range(dim(curr_s)[-1] // self.num_dims)
            #     ])
            # to_add = tmp
            # to_add = curr_s.reshape(-1, curr_s.shape[-1]//self.num_dims, self.num_dims)
    
            for t in range(len(curr_s)): # Per timestep                
                # cost_matrix_for_this_test_timestamp = []
                # for v in range(len(self.X[curr_cycle][t])):
                cost_matrix_for_this_timestamp = np.array([np.abs(np.array(curr_s[t])[x::self.num_dims] - self.X[curr_cycle][:, :, x].T).astype(np.float64).sum(axis=1) for x in range(self.num_dims)]).T
                # print(cost_matrix_for_this_timestamp.shape)
                # quit()
                # cost_matrix_for_this_test_timestamp.append(cost_matrix_for_this_timestamp)
                
                cost_matrix_for_this_test_sample.append(np.array(cost_matrix_for_this_timestamp))
            cost_matrices.append(np.array(cost_matrix_for_this_test_sample))
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
        for i in range(self.num_phases):
            x_new, y_new, q_new = self._create_gallery(i, X, y, q)
            x_vals.append(np.array(x_new))
            y_vals.append(y_new)
            q_vals.append(q_new)
        return x_vals, y_vals, q_vals
    
    
    def _create_gallery(self, step, X, y, q, total_steps=4):
        # Feature Vector for Stage 1: (n_features//3, n_classes * n_samples * n_timesteps, 3)
        first_stages = X[step::total_steps]
        y_possibilities = y[step::total_steps]
        q_for_this_stage = q[step::total_steps]
        set_of_matrices = []
        set_of_qs = None
        y_selection = []
        for i in range(self.dataset.n_classes):
            selection_indices = np.where(y_possibilities == i)[0]
            tmp1 = []
            tmp2 = []
            for s in selection_indices:
                tmp1.append(first_stages[s])
                tmp2.append(q_for_this_stage[s])

            
            curr_selection = tmp1# first_stages[np.where(y_possibilities == i),...]
            curr_q = tmp2 # q_for_this_stage[np.where(y_possibilities == i),...]
            # print(dim(curr_selection))
            # quit()
            tmp = []
            for j in range(len(curr_selection)):
                    tmp.extend(curr_selection[j])
                # for k in range(len(curr_selection[j])):

            curr_selection = tmp

            tmp = []
            for a in curr_q:
                tmp.extend(a)

            # print(dim(curr_q))
            # quit()
            curr_q = tmp#curr_q.reshape(-1)

            if set_of_qs is None:
                set_of_qs = curr_q
            else:
                set_of_qs = np.concatenate((set_of_qs, curr_q))

            for f in range(dim(X)[-1] // self.num_dims):
                # print(curr_selection)
                curr_feature_selection = [x[self.num_dims * f: self.num_dims * f + self.num_dims] for x in curr_selection]
                if len(set_of_matrices) <= f:
                    set_of_matrices.append(curr_feature_selection)
                else:
                    set_of_matrices[f] = np.concatenate((set_of_matrices[f], curr_feature_selection), axis=0)
            y_selection += [i] * len(curr_selection)

        return np.array(set_of_matrices), y_selection, set_of_qs
