import os

import numpy as np
import cv2


from scipy.signal import savgol_filter, find_peaks

from tqdm import tqdm
import matplotlib.pyplot as plt

import mediapipe as mp
mp_pose = mp.solutions.pose

from preprocessing import preprocess_dataset, dist, global_coordinate_frame

def dim(l):
    if not type(l) == list:
        return []
    return [len(l)] + dim(l[0])

def distance(point1, point2):
    x1, y1 = np.array(point1[:][0]), np.array(point1[:][1])
    x2, y2 = np.array(point2[:][0]), np.array(point2[:][1])
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class VideoDataset:

    def __init__(self, label_dir="labels/", info_dir="info/", max_samples = None):
        # skeletons = (video_id, frame_id, kp_id, 2)
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.C_min = 1000000000
        self.C_max = 0
        self.vid_skeletons, self.labels = self._read_skeletons(max_samples)
        self.n_classes = len(np.unique(self.labels))
        self.norm_skeletons = self.normalize_skeletons()
        s = self.norm_skeletons
        # print(len(s[0][0]))
        # quit()
        # self.distances = self.get_distances()
        self.vid_features = self.extract_features()
        v = self.vid_features
        # C_min, C_max = np.min(np.array(self.vid_skeletons).flatten()), np.max(np.array(self.vid_skeletons).flatten())
        self.norm_const = self.C_max - self.C_min

        self.gait_cycles = self.get_gait_cycles()
        self.gait_phases = self.get_gait_phases()
        self.compress_gait_phases()

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

    def extract_features(self):
        features = []
        for i, vid in enumerate(self.norm_skeletons): 
            prev_skel = None
            frame_skels = []
            loop = tqdm(vid)
            for skel in loop:
                frame_skels.append(preprocess_dataset(skel, prev_skel))
                prev_skel = skel
                loop.set_postfix(vid_number=i+1)
            features.append(frame_skels)                
        return features

    def normalize_skeletons(self):
        norm_skel = []
        for vid in self.vid_skeletons:
            vid_skel = []
            for frame in vid:
                point1 = frame[np.argmax(frame[:][1])]
                point2 = frame[np.argmin(frame[:][1])]
                norm_dist = dist(point1, point2)
                norm_kp = []
                for kp in frame:
                    x = 0 if norm_dist == 0 else kp[0] / norm_dist
                    y = 0 if norm_dist == 0 else kp[1] / norm_dist
                    norm_kp.append([x, y])
                vid_skel.append(norm_kp)
            norm_skel.append(vid_skel)
        return norm_skel

    def _read_skeletons(self, max_samples=None):
        skeletons = []
        loop = tqdm(zip(os.listdir(self.label_dir), os.listdir(self.info_dir))) if max_samples is None else tqdm(zip(os.listdir(self.label_dir)[:max_samples], os.listdir(self.info_dir)[:max_samples]))
        i = 1
        labels = []
        for filename, info_name in loop:

            file_kp = self._read_file(self.label_dir + filename, self.info_dir + info_name)
            if file_kp is not None:
                labels.append(filename.split("-")[0])
                skeletons.append(file_kp)
            loop.set_postfix(filename=filename)
        return skeletons, labels

    
    def _read_file(self, filename, info_name):
        frames = []
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
            for frame in kp_arr:
                kps = frame.split(";")
                k_arr = []
                for k in kps:
                    points = k.split(",")
                    p_arr = []
                    for p in points:
                        curr_val = int(float(p))
                        self.C_min = min(curr_val, self.C_min)
                        self.C_max = max(curr_val, self.C_max)
                        p_arr.append(curr_val)
                    k_arr.append(p_arr)
                frames.append(global_coordinate_frame(k_arr))
            return frames


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
            ankle1 = kp[mp_pose.PoseLandmark.LEFT_ANKLE]
            ankle2 = kp[mp_pose.PoseLandmark.RIGHT_ANKLE]
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

class KinectDataset:
    def __init__(self, directory="KinectDataset/"):
        self.directory = directory
        self.filedata = self._get_file_data() # (n_people, n_files, n_lines, 2)

    def _get_file_data(self):
        ret_val = []
        for filename in os.listdir(self.directory):
            curr_person = []
            for f in os.listdir(self.directory + filename):
                with open(self.directory + filename + f) as infile:
                    file_contents = infile.read().split('\n')
                    for x in file_contents:
                        curr_file.append([float(a) for a in x.split(';')[:2]])
                curr_person.append(curr_file)
            ret_val.append(curr_person)
        return ret_val # (n_people, n_files, n_lines, 2)


