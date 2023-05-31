import os

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.interpolate import interp1d

from scipy.signal import savgol_filter, find_peaks, argrelmax, argrelmin

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
np.seterr(all='raise')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.animation as animation
from celluloid import Camera
import mediapipe as mp
mp_pose = mp.solutions.pose

def cell_callback_factory(num_frames):

    def cell_callback(curr_frame, total_frames):
        if curr_frame % (num_frames // 10) == 0:
            print(f"Frame {curr_frame} / {num_frames}")
    return cell_callback


def _dim(l, check_for_error):
    if type(l) != list:
        return []
    else:
        if type(l[0]) == list and check_for_error:
            next_dim = len(l[0])
            for mini_l in l[1:]:
                if len(mini_l) != next_dim:
                    raise ValueError("Array is sparse")
        return [len(l)] + _dim(l[0], check_for_error)

def dim(l, check_for_error=True):
    return tuple(_dim(l, check_for_error))




class GenericGaitDataset:
    """
    Arguments:
    - directory
    - max_samples
    - t_interp
    - num_dims
    - generate_test_video (int - ID of video)

    Need to include:
    - self._get_file_data(max_samples)
        - Gets from self.directory
        - Returns self.skel_data, self.kp_indices
            self.skel_data: Shape (n_people, n_files, n_lines, num_dims)
            self.kp_indices: String => int dictionary for keypoints

    self.setup_information()
        - self.step_classifier (optional)
        - self.connections (list of string two-tuples)
        - self.upper_torso (list of strings defining upper torso)
        - self.lower_torso (list of strings defining lower torso)
    """
    def __init__(self, directory="../KinectDataset/", max_samples=None, t_interp=6, num_dims=3, generate_test_video=None):
        self.num_dims = num_dims
        self.directory = directory
        self.t_interp = t_interp
        self.skel_data, self.kp_indices = self._get_file_data(max_samples) # (n_people, n_files, n_lines, 3)
        self.setup_information()
        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        
        self.X, self.labels = self.reshape_skeletons()

        self.translation_vector()
        self.scaling_vector()
        if generate_test_video is not None:
            self.show_video(generate_test_video)
        if num_dims > 2:
            self.rotation_vector(2)
            if generate_test_video is not None:
                self.show_video(generate_test_video)
        if generate_test_video is not None:
            quit()

        self.X, self.y = self.get_individual_steps()
        
        self.n_classes = len(np.unique(self.y))

        self.interpolate_by_time()

        self.q = self.quality_matrices()

        self.y_raw = self.y.copy()
        self.y = self.to_one_hot()
        self.X = self.get_position_vectors()
        self.split_train_and_test()


    def setup_information(self):
        raise NotImplementedError()


    def train_classifier(self, X):
        loop = tqdm(range(len(X)))
        print("Training classifier")
        train_X_data = []
        train_y_data = []
        for i in loop:
            labels = [x % 4 for x in range(len(X[i]))]
            train_y_data.extend(labels)
            train_X_data.extend(X[i])
        self.step_classifier.fit(train_X_data, train_y_data)
        y_pred = self.step_classifier.predict(train_X_data)
        accuracy = accuracy_score(train_y_data, y_pred)
        precision = precision_score(train_y_data, y_pred, average='macro')
        recall = recall_score(train_y_data, y_pred, average='macro')
        f1 = f1_score(train_y_data, y_pred, average='macro')

        out_str = f"""###############################################
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1: {f1:.3f}
        Accuracy: {accuracy:.3f}

###############################################
        """
        print(out_str)
        quit()

            
            

    def adjust_for_quality(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.X[i, j, :] *= 1 / self.q[i, j]


    def quality_matrices(self):
        def dist(a, b):
            return np.sqrt(np.sum([(x - y)**2 for (x, y) in zip(a, b)]))
        qualities = []
        for i in range(self.X.shape[0]):
            vid_qualities = []
            for j in range(self.X.shape[1]):
                def get_indices(ind_str):
                    return self.X[i, j, self.num_dims * self.kp_indices[ind_str]:self.num_dims * self.kp_indices[ind_str] + self.num_dims]
                
                left_arm = get_indices(self.left_elbow)
                right_arm = get_indices(self.right_elbow)
                left_leg = get_indices(self.left_knee)
                right_leg = get_indices(self.right_knee)

                ct, _, _ = self.get_centroid(self.X[i,j,:])
                qarm = min(dist(ct, left_arm) / dist(ct, right_arm), dist(ct, right_arm) / dist(ct, left_arm))
                qleg = min(dist(ct, left_leg) / dist(ct, right_leg), dist(ct, right_leg) / dist(ct, left_leg))
                vid_qualities.append(min(qleg, qarm))
            qualities.append(vid_qualities)
        return np.array(qualities)



    def get_position_vectors(self):
        joints = [
            self.left_elbow, self.right_elbow, self.left_knee, self.right_knee
        ]
        subset_joints = [
            (self.left_elbow, self.left_wrist),
            (self.right_elbow, self.right_wrist),
            (self.left_knee, self.left_ankle),
            (self.right_knee, self.right_ankle)
        ]
        new_X = []
        for i in range(self.X.shape[0]):
            curr_person = []
            for j in range(self.X.shape[1]):
                ct = self.get_centroid(self.X[i,j,:])[0]
                j_v = []
                for jo in joints:
                    j_coords = self.kp_indices[jo] * self.num_dims
                    curr_j = list(self.X[i,j,j_coords:j_coords + self.num_dims])
                    distances = [(a - b) for (a, b) in zip(curr_j, ct)]
                    # print(type(distances))
                    # print(type(ct))
                    # quit()
                    j_v.extend(list(distances))
                for jo1, jo2 in subset_joints:
                    j_coords1 = self.kp_indices[jo1] * self.num_dims
                    j_coords2 = self.kp_indices[jo2] * self.num_dims
                    
                    curr_j1 = list(self.X[i,j,j_coords1:j_coords1 + self.num_dims])
                    curr_j2 = list(self.X[i,j,j_coords2:j_coords2 + self.num_dims])
                    
                    distances = [(a - b) for (a, b) in zip(curr_j2, curr_j1)]
                    j_v.extend(list(distances))
                    

                
                curr_person.append(j_v)
            new_X.append(curr_person)
        # print(new_X[0][0][0])
        # quit()
        return np.array(new_X, dtype=np.float32)


    def split_train_and_test(self, split=0.1):
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_raw_train, self.y_raw_test, self.q_train, self.q_test = train_test_split(self.X, self.y, self.y_raw, self.q, test_size=split)

    def to_one_hot(self):
        onehot_vector = np.zeros((len(self.y), self.n_classes))
        for i in range(len(self.y)):
            onehot_vector[i, self.y[i]] = 1
        return onehot_vector



    def interpolate_by_time(self, convert_to_numpy=True):
        min_frames = min([len(x) for x in self.X])
        for i in range(len(self.X)):
            curr_sample = self.X[i]
            x = np.arange(len(curr_sample))
            f = interp1d(x, curr_sample, axis=0)
            xnew = np.linspace(0, len(curr_sample) - 1, min_frames)
            # print(xnew, len(curr_sample))
            # quit()
            ynew = f(xnew)
            self.X[i] = ynew
        if convert_to_numpy:
            self.X = np.array(self.X)
        
    # def interpolate_by_time(self):
    #     min_frames = min([len(x) for x in self.sectioned_gaits])
    #     for i in range(len(self.sectioned_gaits)):
    #         curr_sample = self.sectioned_gaits[i]
    #         x = np.arange(len(curr_sample))
    #         f = interp1d(x, curr_sample, axis=0)
    #         xnew = np.linspace(0, len(curr_sample) - 1, min_frames)
    #         ynew = f(xnew)
    #         self.sectioned_gaits[i] = ynew
    #     self.sectioned_gaits = np.array(self.sectioned_gaits)
    def get_individual_steps(self):
        peaks = self.find_peaks()
        ret_val = []
        ret_labels = []
        for i in range(len(self.X)):
            vid_peaks = peaks[i]
            vid_steps = self.X[i]
            for j in range(1, len(vid_peaks[1:])):
                # print(j)
                r = vid_peaks[j] - vid_peaks[j - 1]
                if r <= self.t_interp:
                    continue
                ret_val.append(vid_steps[vid_peaks[j-1]:vid_peaks[j]])
                ret_labels.append(self.labels[i])
        assert len(ret_val) == len(ret_labels)
        return ret_val, ret_labels




    def smooth_walk(self, x, window=4):
        d_trans = []
        for i in range(len(x)):
            lower = 0 if i < window else i - window
            upper = -1 if i + window >= len(x) else i + window
            r = upper - lower
            d_trans.append(sum(x[lower:upper]) / r)
        return d_trans
        

    def _find_peaks_for_video(self, i):
        d = self.smooth_walk(self.ankle_distances(i))
        filtered_distances = savgol_filter(d, 9, 3)
        max_peaks = argrelmax(filtered_distances)[0]
        min_peaks = argrelmin(filtered_distances)[0]
        # print(max_peaks)
        # print(min_peaks)
        peaks = np.sort(np.concatenate([min_peaks, max_peaks]))
        max_peaks_match = False
        test_max = peaks[::2] == max_peaks
        if type(test_max) != bool:
            test_max = test_max.all()
        max_peaks_match = max_peaks_match or test_max
        
        test_max = peaks[1::2] == max_peaks
        if type(test_max) != bool:
            test_max = test_max.all()
        max_peaks_match = max_peaks_match or test_max

        assert max_peaks_match
        

        return peaks
    
    def find_peaks(self):
        peaks = []
        for i in range(len(self.X)):
            peaks.append(self._find_peaks_for_video(i))

        
        return peaks


        

    def show_video(self, i, include_centroids=False, max_frames=200, outfile = "plots.gif"):
        frames = []
        loop = tqdm(range(len(self.X[i][:(-1 if len(self.X[i]) < max_frames else max_frames)])))
        ankles, l, r = self.ankle_distances(i, return_positions=True)
        peaks = self._find_peaks_for_video(i)[::2]
        step_count = 1
        
        fig, ax = plt.subplots(2)
        # ax = fig.add_subplot()
        # ankle_ax = fig.add_subplot()
        ax, ankle_ax = ax[0], ax[1]
        camera = Camera(fig)
        
        print("Generating Video...")
        for a in loop:
            if peaks[(step_count - 1) % len(peaks)] < a < peaks[(step_count) % len(peaks)] :
                step_count += 1
            curr_frame = []
            kps = self.X[i][a]
            ax.legend([f"Step Count: {step_count}"], loc='upper left')
            ankle_ax.plot(ankles[:a])
            if include_centroids:
                
                ct, ut, lt = self.get_centroid(kps)
                xc = [
                    ct[0],
                    ut[0],
                    lt[0]
                ]

                
                yc = [
                    ct[1],
                    ut[1],
                    lt[1]
                ]
              
                a1 = [ct[0] + 1, self.rmov[0] + 1]
                a2 = [ct[1] + 1, self.rmov[1] + 1]
                a3 = [ct[0] + 1, self.rtop[0] + 1]
                a4 = [ct[1] + 1, self.rtop[1] + 1]
                a5 = [ct[0] + 1, self.rleft[0] + 1]
                a6 = [ct[1] + 1, self.rleft[1] + 1]

                ax.scatter(xc, yc)
                ax.scatter(a1, a2)
                ax.plot(a1, a2, c='g')
                ax.scatter(a3, a4)
                ax.plot(a3, a4, c='g')
                ax.scatter(a5, a6)
                ax.plot(a5, a6, c='g')
                ax.plot(xc, yc, c='r')

            coords = [kps[a::self.num_dims] for a in range(self.num_dims)]


            for c in self.connections:
                lines = [[]] * len(coords)

                for a, co in enumerate(coords):
                    c1, c2 = c
                    p1, p2 = co[self.kp_indices[c1]], co[self.kp_indices[c1]]

                    lines[a].append(p1)
                    lines[a].append(p2)

                
                # curr_frame.append(ax.plot(xline, yline, c='b'))
                ax.plot(*lines[:2], c='b')

            
            # curr_frame.append(ax.scatter(x, y))
            # frames.append([ax.scatter(x,y)])
            ax.scatter(*lines[:2])
            camera.snap()
            loop.set_postfix()
        # fig.savefig("3d.png")
        print("Rendering Video")
        # ani = animation.ArtistAnimation(fig, frames, interval=50)
        animation = camera.animate() 
        print("Writing Video")
        animation.save(outfile, writer='imagemagick', progress_callback=cell_callback_factory(len(self.X[i][:(-1 if len(self.X[i]) < max_frames else max_frames)])))
        # ani.save('movie.mp4')


    def ankle_distances(self, i, return_positions=False):
        lankle = self.kp_indices[self.left_ankle] * self.num_dims
        rankle = self.kp_indices[self.right_ankle] * self.num_dims
        ankle_dist = []
        left = []
        right = []
        for j in range(len(self.X[i])):
            ankle1 = self.X[i][j][lankle: lankle + self.num_dims]
            ankle2 = self.X[i][j][rankle: rankle + self.num_dims]
            d = np.sqrt(sum([(a - b)**2 for a, b in zip(ankle1,ankle2)]))
            ankle_dist.append(d)
            left.append(ankle1[-1])
            right.append(ankle2[-1])
        if return_positions:
            return ankle_dist, left, right

        return ankle_dist



    def scaling_vector(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                ct, ut, lt = self.get_centroid(self.X[i][j])
                self.X[i][j] = self._scaling_vector(self.X[i][j], ut, lt)

    def _scaling_vector(self, X, ut, lt):
        diff = [a - b for (a, b) in zip(ut, lt)]
        scale_val = np.sqrt(sum([a**2 for a in diff]))
        return [a / scale_val for a in X]



        
    def get_centroid(self, X):
        # Step 1: Get upper Centroid and Lower Centroid
        upper_limbs = []
        for u in self.upper_torso:
            start_ind = self.num_dims * self.kp_indices[u]
            upper_limbs.extend(X[start_ind: start_ind + self.num_dims])
        uc = [sum(upper_limbs[x::self.num_dims]) / len(self.upper_torso) for x in range(self.num_dims)]

        lower_limbs = []
        for l in self.lower_torso:
            start_ind = self.num_dims * self.kp_indices[l]
            lower_limbs.extend(X[start_ind: start_ind + self.num_dims])
        lc = [sum(lower_limbs[x::self.num_dims]) / len(self.lower_torso) for x in range(self.num_dims)]

        return np.array([(x + y) / 2 for (x, y) in zip(uc, lc)]), uc, lc

    def rotation_vector(self, tm):
        print("Creating Rotation Matrix")
        loop = tqdm(range(len(self.X)))
        for i in loop:
            
            centroids = []
            for j in range(len(self.X[i])):                
                centroids.append( self.get_centroid(self.X[i][j]))

                ct, ut, lt = centroids[-1]
                if j > tm:
                    ang = 0
                    ct_prev, _,_ = centroids[j - tm]
                    self.X[i][j], ang = self._rotation_vector(self.X[i][j], self.X[i][j - tm], ct, ut, lt, ct_prev, ang)

    def _rotation_vector(self, X, Xp, ct, ut, lt, ct_prev, def_ang):
        d = np.sqrt(sum([(a - b)**2 for a, b in zip(ct, ct_prev)]))
        # try:
        # rmov = []
        # for a, b in zip(ct, ct_prev):
        #     try:
        #         rmov.append((a - b) / d)
        #     except FloatingPointError:
        #         print(a, b, (a - b), d)
        #         quit()
        if d > 0:
            rmov = [(a - b) / d for a, b in zip(ct, ct_prev)]
            # except FloatingPointError:
                # print(a, b, (a - b),)
            self.rmov = rmov
            x = np.abs(rmov[0] - ct[0])
            r = np.sqrt((rmov[0] - ct[0]) ** 2 + (rmov[2] - ct[2]) ** 2)
            ang = np.arcsin(x / r) if r > 0 else 0
        else:
            ang = def_ang

        R_inv = [
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)]
        ]

        # dcen = np.sqrt(sum([(a - b)**2 for a, b in zip(ut, lt)]))
        # rtop = [(a - b) / dcen for a, b in zip(ut, lt)]
        # self.rtop = rtop

        # cross_prod = np.cross(rtop, rmov)
        # dcross = np.sqrt(sum(a**2 for a in cross_prod))
        # rleft = [a / dcross for a in cross_prod]
        # self.rleft = rleft

        # R_mat = np.array([
        #     rmov,
        #     rtop,
        #     rleft
        # ])

        # R_inv = R_mat # np.linalg.inv(R_mat)

        # print(R_inv.shape)
        # quit()

        ret_val = []
        for i in range(len(X) // self.num_dims):
            curr_vals = X[i * self.num_dims: i * self.num_dims + self.num_dims]
            # print(np.dot(R_inv, curr_vals).shape)
            # quit()
            ret_val += list(np.dot(R_inv, curr_vals))

        return ret_val, ang


        # cross_prod = np.cross(rtop, rmov)
        

    def translation_vector(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                ct, ut, lt = self.get_centroid(self.X[i][j])
                self.X[i][j] = self._translation_vector(self.X[i][j], ct)

    def _translation_vector(self, X, ct):        
        # Step 2: Translation Vector
        pt = ct.T
        for i in range(self.num_dims):
            X[i::self.num_dims] -= pt[i]

        return X


    def draw_keypoints(self, i, j, num_frames=1, include_centroids=False):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for a in range(num_frames):
            kps = self.X[i][j + a]
            ct, ut, lt = self.get_centroid(kps)
            if include_centroids:
                xc = [
                    ct[0],
                    ut[0],
                    lt[0]
                ]

                
                yc = [
                    ct[1],
                    ut[1],
                    lt[1]
                ]
                


                zc = [
                    ct[2],
                    ut[2],
                    lt[2]
                ]

                
                ax.scatter3D(xc, yc, zc)
                ax.plot3D(xc, yc, zc, c='r')

            x = kps[0::3]
            y = kps[1::3]
            z = kps[2::3]

            xline = []
            yline = []
            zline = []

            for c in self.connections[:]:
                c1, c2 = c
                x1 = x[self.kp_indices[c1]]
                x2 = x[self.kp_indices[c2]]
                
                y1 = y[self.kp_indices[c1]]
                y2 = y[self.kp_indices[c2]]
                
                z1 = z[self.kp_indices[c1]]
                z2 = z[self.kp_indices[c2]]

                xline.append(x1)
                xline.append(x2)

                
                yline.append(y1)
                yline.append(y2)

                
                zline.append(z1)
                zline.append(z2)

                
                ax.plot3D(xline, yline, zline, c='b')
                
                xline = []
                yline = []
                zline = []

            ax.scatter3D(x, y, z)
        # fig.savefig("3d.png")
        # plt.show()

    def _get_file_data(self, max_samples):
        raise NotImplementedError()

    def reshape_skeletons(self):
        # Need skeleton to be shape (vid_seq, frame, keypoints, 2)
        labels = []
        reshaped_skel = []
        for i, person in enumerate(self.skel_data):
            # reshaped_skel.append([])
            for f in person:
                reshaped_skel.append([])
                labels.append(i)
                for l in f:
                    if l[0] == self.headpoint:
                        reshaped_skel[-1].append([])
                    reshaped_skel[-1][-1] += l[1:]
                if len(reshaped_skel[-1][-1]) % 20 != 0:
                    print(len(reshaped_skel[-1][-1]))
                    quit()
        return reshaped_skel, labels
