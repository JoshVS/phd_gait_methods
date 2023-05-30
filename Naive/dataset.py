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




class NaiveKinectDataset:
    def __init__(self, directory="../KinectDataset/", max_samples=None, t_interp=6):
        self.directory = directory
        self.t_interp = t_interp
        self.skel_data, self.kp_indices = self._get_file_data(max_samples) # (n_people, n_files, n_lines, 3)
        # print(dim(self.skel_data, check_for_error=False))
        # quit()
        self.step_classifier = RandomForestClassifier()
        
        self.connections = [
            ('Head', 'Shoulder-Center'),
            ('Shoulder-Center', 'Shoulder-Right'),
            ('Shoulder-Center', 'Shoulder-Left'),
            ('Shoulder-Center', 'Spine'),
            ('Spine', 'Hip-centro'),
            ('Hip-centro', 'Hip-Left'),            
            ('Hip-centro', 'Hip-Right'),            
            ('Hip-Right', 'Knee-Right'),            
            ('Hip-Left', 'Knee-Left'),            
            ('Knee-Right', "Ankle-Right"),                    
            ("Ankle-Right", 'Foot-Right'),        
            ('Knee-Left', "Ankle-Left"),                    
            ("Ankle-Left", 'Foot-Left'),
            ("Shoulder-Left", "Elbow-Left"),
            ("Elbow-Left", "Wrist-Left"),
            ("Wrist-Left", "Hand-Left"),
            ("Shoulder-Right", "Elbow-Right"),
            ("Elbow-Right", "Wrist-Right"),
            ("Wrist-Right", "Hand-Right"),
        ]
        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        self.vid_skeletons, self.labels = self.reshape_skeletons()
        self.X = self.vid_skeletons
        self.translation_vector()
        self.scaling_vector()
        # before_rotation = self.X.copy()
        test_ankles = []
        for i in range(len(self.X)):
            test_ankles.append(self.ankle_distances(i))
        # self.show_video(2, include_centroids=False, max_frames = 200)
        self.rotation_vector(2)
        # self.show_video(2, include_centroids=False, max_frames = 200, outfile="rotated.gif")
        for i in range(len(self.X)):
            curr_ank = self.ankle_distances(i)
            test = [a - b for a, b in zip(curr_ank, test_ankles[i])]
            test_bool = [a > 15 for a in test]
            if sum(test_bool) > 0:
                print("ERROR")
                print(max(test))
                quit()
            # assert self.ankle_distances(i) == test_ankles[i]
        # quit()
        self.X, self.y = self.get_individual_steps()
        # print(dim(self.X, check_for_error=False))
        # quit()
        # self.train_classifier(self.X)
        self.n_classes = len(np.unique(self.y))
        self.interpolate_by_time()
        self.q = self.quality_matrices()
        self.adjust_for_quality()
        self.y_raw = self.y.copy()
        self.y = self.to_one_hot()
        self.X = self.get_position_vectors()
        self.split_train_and_test()


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
                    return self.X[i, j, 3 * self.kp_indices[ind_str]:3 * self.kp_indices[ind_str] + 3]
                
                left_arm = get_indices("Elbow-Left")
                right_arm = get_indices("Elbow-Right")
                left_leg = get_indices("Knee-Left")
                right_leg = get_indices("Knee-Right")

                ct, _, _ = self.get_centroid(self.X[i,j,:])
                qarm = min(dist(ct, left_arm) / dist(ct, right_arm), dist(ct, right_arm) / dist(ct, left_arm))
                qleg = min(dist(ct, left_leg) / dist(ct, right_leg), dist(ct, right_leg) / dist(ct, left_leg))
                vid_qualities.append(min(qleg, qarm))
            qualities.append(vid_qualities)
        return np.array(qualities)



    def get_position_vectors(self):
        joints = [
            "Elbow-Left", "Elbow-Right", "Knee-Left", "Knee-Right"
        ]
        subset_joints = [
            ("Elbow-Left", "Wrist-Left"),
            ("Elbow-Right", "Wrist-Right"),
            ("Knee-Left", "Ankle-Left"),
            ("Knee-Right", "Ankle-Right")
        ]
        new_X = []
        for i in range(self.X.shape[0]):
            curr_person = []
            for j in range(self.X.shape[1]):
                ct = self.get_centroid(self.X[i,j,:])[0]
                j_v = []
                for jo in joints:
                    j_coords = self.kp_indices[jo] * 3
                    curr_j = list(self.X[i,j,j_coords:j_coords + 3])
                    distances = [(a - b) for (a, b) in zip(curr_j, ct)]
                    # print(type(distances))
                    # print(type(ct))
                    # quit()
                    j_v.extend(list(distances))
                for jo1, jo2 in subset_joints:
                    j_coords1 = self.kp_indices[jo1] * 3
                    j_coords2 = self.kp_indices[jo2] * 3
                    
                    curr_j1 = list(self.X[i,j,j_coords1:j_coords1 + 3])
                    curr_j2 = list(self.X[i,j,j_coords2:j_coords2 + 3])
                    
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
                


                zc = [
                    ct[2],
                    ut[2],
                    lt[2]
                ]

                a1 = [ct[0] + 1, self.rmov[0] + 1]
                a2 = [ct[1] + 1, self.rmov[1] + 1]
                a3 = [ct[0] + 1, self.rtop[0] + 1]
                a4 = [ct[1] + 1, self.rtop[1] + 1]
                a5 = [ct[0] + 1, self.rleft[0] + 1]
                a6 = [ct[1] + 1, self.rleft[1] + 1]

                
                # curr_frame.append(ax.scatter(xc, yc))
                # curr_frame.append(ax.plot(xc, yc, c='r'))
                ax.scatter(xc, yc)
                ax.scatter(a1, a2)
                ax.plot(a1, a2, c='g')
                ax.scatter(a3, a4)
                ax.plot(a3, a4, c='g')
                ax.scatter(a5, a6)
                ax.plot(a5, a6, c='g')
                ax.plot(xc, yc, c='r')

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

                
                # curr_frame.append(ax.plot(xline, yline, c='b'))
                ax.plot(xline, yline, c='b')
                
                xline = []
                yline = []
                zline = []

            
            # curr_frame.append(ax.scatter(x, y))
            # frames.append([ax.scatter(x,y)])
            ax.scatter(x,y)
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
        lankle = self.kp_indices['Ankle-Left'] * 3
        rankle = self.kp_indices['Ankle-Right'] * 3
        ankle_dist = []
        left = []
        right = []
        for j in range(len(self.X[i])):
            ankle1 = self.X[i][j][lankle: lankle + 3]
            ankle2 = self.X[i][j][rankle: rankle + 3]
            d = np.sqrt(sum([(a - b)**2 for a, b in zip(ankle1,ankle2)]))
            ankle_dist.append(d)
            left.append(ankle1[2])
            right.append(ankle2[2])
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
        upper_limbs = X[:12]
        uc = [
            sum(upper_limbs[0::3]) / 4,
            sum(upper_limbs[1::3]) / 4,
            sum(upper_limbs[2::3]) / 4
        ]
        lower_limbs = X[11 * 3:11 * 3 +12]
        lc = [
            sum(lower_limbs[0::3]) / 4,
            sum(lower_limbs[1::3]) / 4,
            sum(lower_limbs[2::3]) / 4
        ]

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
                    ct_prev, _,_ = centroids[j - tm]
                    self.X[i][j] = self._rotation_vector(self.X[i][j], self.X[i][j - tm], ct, ut, lt, ct_prev)

    def _rotation_vector(self, X, Xp, ct, ut, lt, ct_prev):
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
            x = np.abs(rmov[2] - ct[2])
            r = np.sqrt((rmov[0] - ct[0]) ** 2 + (rmov[2] - ct[2]) ** 2)
            ang = np.arcsin(x / r) if r > 0 else 0

            R_inv = [
                [np.cos(ang), 0, -np.sin(ang)],
                [0, 1, 0],
                [np.sin(ang), 0, np.cos(ang)]
            ]
        else:
            R_inv = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
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
        for i in range(len(X) // 3):
            curr_vals = X[i * 3: i * 3 + 3]
            # print(np.dot(R_inv, curr_vals).shape)
            # quit()
            ret_val += list(np.dot(R_inv, curr_vals))

        return ret_val


        # cross_prod = np.cross(rtop, rmov)
        

    def translation_vector(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                ct, ut, lt = self.get_centroid(self.X[i][j])
                self.X[i][j] = self._translation_vector(self.X[i][j], ct)

    def _translation_vector(self, X, ct):        
        # Step 2: Translation Vector
        pt = ct.T
        X[0::3] -= pt[0]
        X[1::3] -= pt[1]
        X[2::3] -= pt[2]

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
        kp_indices = {}
        ret_val = []
        if max_samples is None:
            loop = tqdm(os.listdir(self.directory))
        elif 0 < max_samples < 1:
            max_ind = int(len(os.listdir(self.directory)) * max_samples)
            loop = tqdm(os.listdir(self.directory)[:max_ind])
        else:
            loop = tqdm(os.listdir(self.directory)[:max_samples])            
        for filename in loop:
            curr_person = []
            for f in os.listdir(self.directory + filename):
                curr_file = []
                with open(self.directory + filename + "/" +  f) as infile:
                    file_contents = infile.read().split('\n')
                    for i, x in enumerate(file_contents):
                        if x == '':
                            continue
                        z = float(x.split(';')[-1])
                        if x.split(';')[0] not in kp_indices.keys():
                            kp_indices[x.split(';')[0]] = i % 20
                        elif kp_indices[x.split(';')[0]] != i % 20:
                            print(f"Whoops {i} != {kp_indices[x.split(';')[0]]}")
                            quit()
                        curr_file.append([x.split(';')[0]] + [float(a) for a in x.split(';')[1:]])
                curr_person.append(curr_file)
            ret_val.append(curr_person)
            loop.set_postfix()
        return ret_val, kp_indices # (n_people, n_files, n_lines, 4)

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
                    if l[0] == "Head":
                        reshaped_skel[-1].append([])
                    reshaped_skel[-1][-1] += l[1:]
                if len(reshaped_skel[-1][-1]) % 20 != 0:
                    print(len(reshaped_skel[-1][-1]))
                    quit()
        return reshaped_skel, labels



    def normalize_skeletons(self):
        norm_skel = []
        for vid in self.vid_skeletons:
            vid_skel = []
            for frame in vid:
                point1 = frame[np.argmax([f[1] for f in frame])]
                point2 = frame[np.argmin([f[1] for f in frame])]
                norm_dist = dist(point1, point2)
                norm_kp = []
                for kp in frame:
                    x = 0 if norm_dist == 0 else kp[0] / norm_dist
                    y = 0 if norm_dist == 0 else kp[1] / norm_dist
                    norm_kp.append([x, y])
                vid_skel.append(norm_kp)
            norm_skel.append(vid_skel)
        return norm_skel

    def show_person(self, p=0, f=0):
        
        person = self.skel_data[p][f] # (n_lines, 3)
        assert person[0][0] == 'Head'
        person_skeleton = [person[0][:]]
        kp_dict = {"Head": person[0][1:]}
        for l in person[1:]: # l = (3,)
            if l[0] == 'Head':
                break
            person_skeleton.append(l[:])
            kp_dict[l[0]] = l[1:]


        plt.figure()
        plt.scatter([p[1] for p in person_skeleton], [p[2] for p in person_skeleton])
        for p in person_skeleton:
            plt.text(p[1], p[2], p[0])
        for from_val, to_val in self.connections:
            plt.plot(
                [kp_dict[from_val][0], kp_dict[to_val][0]],
                [kp_dict[from_val][1], kp_dict[to_val][1]])

        
        

        plt.savefig('output.png')
        plt.close()

    def filter_peaks(self, peaks, threshold=5):
        prev_peak = 0
        new_peaks = []
        for p in peaks:
            if p - prev_peak > threshold:
                new_peaks.append(p)
            prev_peak = p
        return new_peaks

    def get_vid_peaks(self, norm_skel_vid):
        distances = []
        for i, kp in enumerate(norm_skel_vid):
            ankle1 = kp[self.kp_indices["Ankle-Left"]]
            ankle2 = kp[self.kp_indices["Ankle-Right"]]
            distances.append(dist(ankle1, ankle2))
        filtered_distances = savgol_filter(distances, 9, 3)
        peaks = find_peaks(filtered_distances)[0]
        peaks = self.filter_peaks(peaks)
        return peaks, [filtered_distances[x] for x in peaks], filtered_distances

    def get_peaks(self, show_peaks=None):
        vid_peaks = []
        for i, v in enumerate(self.norm_skeletons):
            p, f, d = self.get_vid_peaks(v)
            if i == show_peaks:
                plt.figure()
                plt.plot(d)
                plt.scatter(p, f)
                plt.savefig("peaks.png")
                plt.close()
            vid_peaks.append(p)
        return vid_peaks

    def get_gait_cycles(self):
        """
        Gait cycles happen every second time ankle distances peak
        """
        peaks = self.get_peaks()
        g_arr = []
        for p in peaks:
            g_arr.append(p[::2])
        return g_arr

    def show_gait_cycle(self, vid_seq):
        peaks, peak_vals, distances = self.get_peaks(vid_seq)
        plt.plot(distances)
        plt.scatter(peaks, peak_vals)
        plt.savefig("output.png")
        plt.close()

    def compress_gait_phases(self):
        ret_val = []
        label_ret = []
        for sample, label in zip(self.gait_phases, self.labels):
            ret_val.extend(sample)
            label_ret.extend([label] * len(sample))
        self.labels = label_ret
        self.gait_phases = ret_val


    def get_gait_phases(self):
        gait_phases = [
            (0, 10),
            (10, 30),
            (30, 50),
            (50, 60),
            (60, 73),
            (73, 87),
            (87, 100)
        ]
        final_gait_features = []
        for i, v in enumerate(self.gait_cycles):
            vid_g_f = []
            if len(v) == 0:
                self.labels.pop(i)
                continue
            prev_g = v[0]
            for g in v[1:]:
                len_gait = g - prev_g
                curr_gp = []
                for gp in gait_phases:
                    start, stop = gp
                    start = prev_g + int(start / 100 * len_gait)
                    stop = prev_g + int(stop / 100 * len_gait)
                    curr_f = []
                    for f in range(5):
                        curr_vid = self.vid_features[i]
                        curr_frames = curr_vid[start:stop]
                        tmp = None
                        for fr in curr_frames:
                            tmp = fr[f] if tmp is None else [x + y for x, y in zip(fr[f], tmp)]
                        tmp = np.divide(tmp, stop - start)
                        curr_f.extend(tmp)
                    
                    curr_gp.append(curr_f)
                vid_g_f.append(curr_gp)
                prev_g = g
            final_gait_features.append(vid_g_f)
        return final_gait_features

    def show_gait_cycle(self, vid_seq=None):
        if vid_seq is None:
            vid_seq = self.vid_skeletons[0]
        self.get_peaks(1)

    def one_hot_labels(self):
        un_labels = list(np.unique(self.labels))
        self.n_classes = len(un_labels)
        onehot = np.zeros((len(self.labels), len(un_labels)))
        for i, l in enumerate(self.labels):
            l_ind = un_labels.index(l)
            onehot[i, l_ind] = 1
        return onehot