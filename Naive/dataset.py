import os

import numpy as np
import cv2


from scipy.signal import savgol_filter, find_peaks

from tqdm import tqdm
import matplotlib.pyplot as plt

import mediapipe as mp
mp_pose = mp.solutions.pose


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

def distance(point1, point2):
    x1, y1 = np.array([p[0] for p in point1]), np.array([p[1] for p in point1])
    x2, y2 = np.array([p[0] for p in point2]), np.array([p[1] for p in point2])
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class NaiveVideoDataset:

    def __init__(self, label_dir="../labels/", info_dir="../info/", max_samples = None):
        # skeletons = (video_id, frame_id, kp_id, 2)
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.vid_skeletons, self.labels = self._read_skeletons(max_samples)
        self.n_classes = len(np.unique(self.labels))

    def shape(self):
        return dim(self.vid_skeletons)

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
            skeletons[i] = skeletons[i][:min_frames]
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


    def filter_peaks(self, peaks, threshold=5):
        prev_peak = 0
        new_peaks = []
        for p in peaks:
            if p - prev_peak > threshold:
                new_peaks.append(p)
            prev_peak = p
        return new_peaks


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
