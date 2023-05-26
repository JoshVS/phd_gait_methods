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

def filter_peaks(peaks, threshold=5):
    prev_peaks = 0
    new_peaks = []
    for p in peaks:
        if p - prev_peaks > threshold:
            new_peaks.append(p)
        prev_peaks = p
    return new_peaks

def distance(point1, point2):
    x1 = point1[:, 0]
    x2 = point1[:, 1]
    y1 = point2[:, 0]
    y2 = point2[:, 1]

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class NaiveVideoDataset:
    """
    X: Shape (n_samples, n_timesteps, n_features)
    """

    def __init__(self, label_dir="../labels/", info_dir="../info/", max_samples = None):
        # skeletons = (video_id, frame_id, kp_id, 2)
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.vid_skeletons, self.labels = self._read_skeletons(max_samples)
        self.vid_skeletons = np.array(self.vid_skeletons, dtype=np.float32)
        self.labels = self.one_hot_labels()
        self.n_classes = self.labels.shape[-1]
        self.X, self.y = shuffle(self.vid_skeletons, self.labels)
        self.normalize_skeletons()
        self.global_coords()
        self.accumulated_frame_difference_energy_image()
        self.distances = self.get_ankle_distances()
        self.sectioned_gaits, self.labels = self.get_gait_cycles()
        self.interpolate_by_time()
        self.X, self.y = self.sectioned_gaits, self.labels

    def interpolate_by_time(self):
        min_frames = min([len(x) for x in self.sectioned_gaits])
        for i in range(len(self.sectioned_gaits)):
            curr_sample = self.sectioned_gaits[i]
            x = np.arange(len(curr_sample))
            f = interp1d(x, curr_sample, axis=0)
            xnew = np.linspace(0, len(curr_sample) - 1, min_frames)
            ynew = f(xnew)
            self.sectioned_gaits[i] = ynew
        self.sectioned_gaits = np.array(self.sectioned_gaits)

        # for i in range(len(skeletons)):
        #     curr_vid = skeletons[i]
        #     x = np.arange(0, len(curr_vid))
        #     f = interp1d(x, curr_vid, axis=0)
        #     xnew = np.linspace(0, len(curr_vid) - 1, min_frames)
        #     ynew = f(xnew)
        #     skeletons[i] = ynew
    def get_gait_cycles(self):
        peaks = self.get_gait_indices()
        ret_val = []
        new_labels = []
        for i in range(len(peaks)):
            curr_p = peaks[i][::2]
            for j in range(1, len(curr_p)):
                ret_val.append(self.raw_data[i][curr_p[j - 1]:curr_p[j]])
                new_labels.append(self.labels[i])
        return ret_val, np.array(new_labels)





    def get_gait_indices(self):
        filtered_peaks = []
        for a in self.distances:
            filtered_distances = savgol_filter(a, 9, 3)
            peaks = find_peaks(filtered_distances)[0]
            filtered_peaks.append(filter_peaks(peaks))
        return filtered_peaks


    def get_ankle_distances(self):
        distances = []
        for i in range(len(self.raw_data)):
            curr_dist = []
            for j in range(len(self.raw_data[i])):
                x1 = self.raw_data[i][j][mp_pose.PoseLandmark.LEFT_ANKLE * 2]                
                y1 = self.raw_data[i][j][mp_pose.PoseLandmark.LEFT_ANKLE * 2 + 1]
                x2 = self.raw_data[i][j][mp_pose.PoseLandmark.RIGHT_ANKLE * 2]
                y2 = self.raw_data[i][j][mp_pose.PoseLandmark.RIGHT_ANKLE * 2 + 1]
                curr_dist.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            distances.append(curr_dist)

        return distances
                
                           

    def accumulated_frame_difference_energy_image(self):
        for i in range(self.X.shape[0]):
            for j in range(1, self.X.shape[1]):
                for k in range(self.X.shape[2]):
                    forward_frame_difference = 0 if self.X[i,j,k] <= self.X[i,j - 1,k] else self.X[i,j,k] - self.X[i,j - 1,k]
                    backward_frame_difference = 0 if self.X[i,j,k] >= self.X[i,j - 1,k] else self.X[i,j - 1,k] - self.X[i,j,k]
                    self.X[i,j,k] = forward_frame_difference + backward_frame_difference


    def global_coords(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.X[i,j,::2] -= self.X[i,j,0]
                self.X[i,j,1::2] -= self.X[i,j,1]
        

    def normalize_skeletons(self):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                max_val = np.max(self.X[i,j,...])
                min_val = np.min(self.X[i,j,...])
                val_range = (max_val - min_val)
                self.X[i,j,...] = (max_val - self.X[i,j,...]) / val_range if val_range != 0  else max_val

    def shape(self):
        return dim(self.vid_skeletons)
    
    def yshape(self):
        return dim(self.labels)

    def one_hot_labels(self):
        un_labels = list(np.unique(self.labels))
        self.n_classes = len(un_labels)
        onehot = np.zeros((len(self.labels), len(un_labels)))
        for i, l in enumerate(self.labels):
            l_ind = un_labels.index(l)
            onehot[i, l_ind] = 1
        return onehot


    def compress_gait_phases(self):
        ret_val = []
        label_ret = []
        for sample, label in zip(self.gait_phases, self.labels):
            ret_val.extend(sample)
            label_ret.extend([label] * len(sample))
        self.labels = label_ret
        self.gait_phases = ret_val

    def get_distances(self):
        dist = []
        for i, point in enumerate(self.vid_skeletons[0][1:]):
            dist.append(distance(point, self.vid_skeletons[0][i]))
        return dist


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



    def _read_skeletons(self, max_samples=None):
        skeletons = []
        loop = tqdm(zip(os.listdir(self.label_dir), os.listdir(self.info_dir))) if max_samples is None else tqdm(zip(os.listdir(self.label_dir)[:max_samples], os.listdir(self.info_dir)[:max_samples]))
        i = 1
        labels = []
        min_frames = None
        for filename, info_name in loop:

            file_kp = self._read_file(self.label_dir + filename, self.info_dir + info_name)
            curr_num_frames = len(file_kp)
            if min_frames is None or curr_num_frames < min_frames:
                min_frames = curr_num_frames
            if file_kp is not None:
                labels.append(filename.split("-")[0])
                skeletons.append(file_kp)
            loop.set_postfix(filename=filename)


        for i in range(len(skeletons)):
            for j in range(len(skeletons[i])):
                new_arr = []
                for k in range(len(skeletons[i][j])):
                    new_arr.append(skeletons[i][j][k][0])
                    new_arr.append(skeletons[i][j][k][1])
                skeletons[i][j] = new_arr
        
        self.raw_data = skeletons.copy()
        
        # Interpolate Frames
        for i in range(len(skeletons)):
            curr_vid = skeletons[i]
            x = np.arange(0, len(curr_vid))
            f = interp1d(x, curr_vid, axis=0)
            xnew = np.linspace(0, len(curr_vid) - 1, min_frames)
            ynew = f(xnew)
            skeletons[i] = ynew

        
        return skeletons, labels

    
    def _read_file(self, filename, info_name):
        frames = []
        min_frames = None
        with open(info_name, "r") as info_file:
            info_text = info_file.read().strip()
            info_arr = info_text.split(";")
            info_dict = {}
            for inf in info_arr:
                k, v = inf.split(",")
                v = int(v)
                info_dict[k] = v

        with open(filename, "r") as kp_file:
            kp_txt = kp_file.read()
            if kp_txt == '':
                return None
            kp_arr = kp_txt.strip().split("\n")
            num_curr_frames = 0
            for frame in kp_arr:
                kps = frame.split(";")
                k_arr = []
                num_curr_frames += 1
                for k in kps:
                    points = k.split(",")
                    p_arr = []
                    for p in points:
                        curr_val = int(float(p))
                        p_arr.append(curr_val)
                    k_arr.append(p_arr)
                frames.append(k_arr)
                if min_frames is None or num_curr_frames < min_frames:
                    min_frames = num_curr_frames
            
            return frames

class NaiveKinectDataset:
    def __init__(self, directory="../KinectDataset/", max_samples=None, t_interp=6):
        self.directory = directory
        self.t_interp = t_interp
        self.skel_data, self.kp_indices = self._get_file_data(max_samples) # (n_people, n_files, n_lines, 3)
        # print(dim(self.skel_data, check_for_error=False))
        # quit()
        
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
        self.X, self.y = self.get_individual_steps()
        self.n_classes = len(np.unique(self.y))
        self.interpolate_by_time()
        self.q = self.quality_matrices()
        self.adjust_for_quality()
        self.y_raw = self.y.copy()
        self.y = self.to_one_hot()
        self.X = self.get_position_vectors()
        self.split_train_and_test()


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


        

    def show_video(self, i, include_centroids=False):
        frames = []
        loop = tqdm(range(len(self.X[i])))
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
            ax.legend([f"Step Count: {step_count}"])
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

                
                # curr_frame.append(ax.scatter(xc, yc))
                # curr_frame.append(ax.plot(xc, yc, c='r'))
                ax.scatter(xc, yc)
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
        animation.save("plots.gif", writer='imagemagick', progress_callback=cell_callback_factory(len(self.X[i])))
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
        for i in range(len(self.X)):
            
            centroids = []
            for j in range(len(self.X[i])):                
                centroids.append( self.get_centroid(self.X[i][j]))

                ct, ut, lt = centroids[-1]
                if j > tm:
                    ct_prev, _,_ = centroids[-1 - tm]
                    self.X[i][j] = self._rotation_vector(self.X[i][j], self.X[i][j - tm], ct, ut, lt, ct_prev)

    def _rotation_vector(self, X, Xp, ct, ut, lt, ct_prev):
        d = np.sqrt(sum([(a - b)**2 for a, b in zip(ct, ct_prev)]))
        rmov = [(a - b) / d for a, b in zip(ct, ct_prev)]

        dcen = np.sqrt(sum([(a - b)**2 for a, b in zip(ut, lt)]))
        rtop = [(a - b) / dcen for a, b in zip(ut, lt)]

        cross_prod = np.cross(rtop, rmov)
        dcross = np.sqrt(sum(a**2 for a in cross_prod))
        rleft = [a / dcross for a in cross_prod]

        R_mat = np.array([
            rmov, 
            rleft, 
            rtop
        ])
        R_inv = np.linalg.inv(R_mat)

        ret_val = []
        for i in range(len(X) // 3):
            curr_vals = X[i * 3: i * 3 + 3]
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