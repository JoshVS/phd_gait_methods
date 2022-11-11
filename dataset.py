import os

import numpy as np
import cv2


from scipy.signal import savgol_filter, find_peaks

from tqdm import tqdm
import matplotlib.pyplot as plt

import mediapipe as mp
mp_pose = mp.solutions.pose

from preprocessing import preprocess_dataset, dist, global_coordinate_frame

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
        self.vid_skeletons = self._read_skeletons(max_samples)
        # self.distances = self.get_distances()
        self.vid_features = self.extract_features()
        # C_min, C_max = np.min(np.array(self.vid_skeletons).flatten()), np.max(np.array(self.vid_skeletons).flatten())
        self.norm_const = self.C_max - self.C_min

        self.norm_skeletons = self.normalize_skeletons()
        self.gait_cycles = self.get_gait_cycles()

    def get_distances(self):
        dist = []
        for i, point in enumerate(self.vid_skeletons[0][1:]):
            dist.append(distance(point, self.vid_skeletons[0][i]))
        return dist



    def extract_features(self):
        features = []
        for i, vid in enumerate(self.vid_skeletons): 
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


    # def normalize_skeletons(self):
    #     norm_skel = []
    #     for vid in self.vid_skeletons:
    #         vid_skel = []
    #         for frame in vid:
    #             norm_kp = []
    #             for kp in frame:
    #                 x = (kp[0] - self.C_min) / self.norm_const * 10
    #                 y = (kp[1] - self.C_min) / self.norm_const * 10
    #                 norm_kp.append([x, y])
    #             vid_skel.append(norm_kp)
    #         norm_skel.append(vid_skel)
    #     return norm_skel

    def _read_skeletons(self, max_samples=None):
        skeletons = []
        loop = tqdm(zip(os.listdir(self.label_dir), os.listdir(self.info_dir))) if max_samples is None else tqdm(zip(os.listdir(self.label_dir)[:max_samples], os.listdir(self.info_dir)[:max_samples]))
        i = 1
        for filename, info_name in loop:
            file_kp = self._read_file(self.label_dir + filename, self.info_dir + info_name)
            if file_kp is not None:
                skeletons.append(self._read_file(self.label_dir + filename, self.info_dir + info_name))
            loop.set_postfix(filename=filename)
        # print(len(skeletons[0][0]))
        # quit()
        return skeletons

    
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


    def filter_peaks(peaks, threshold=5):
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

    def get_peaks(self):
        vid_peaks = []
        for v in self.norm_skeletons:
            vid_peaks.append(self.get_vid_peaks(v))
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
        quit()