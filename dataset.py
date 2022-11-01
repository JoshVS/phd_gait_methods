import os

import numpy as np
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt

import mediapipe as mp
mp_pose = mp.solutions.pose

from preprocessing import preprocess_dataset

def distance(point1, point2):
    x1, y1 = np.array(point1[:][0]), np.array(point1[:][1])
    x2, y2 = np.array(point2[:][0]), np.array(point2[:][1])
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class VideoDataset:

    def __init__(self, label_dir="labels/", info_dir="info/"):
        # skeletons = (video_id, frame_id, kp_id, 2)
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.vid_skeletons = self._read_skeletons()
        self.distances = self.get_distances()
        self.video_features = self.extract_features()
        C_min, C_max = min(self.vid_skeletons), max(self.vid_skeletons)

    def get_distances(self):
        dist = []
        for i, point in enumerate(self.vid_skeletons[0][1:]):
            dist.append(distance(point, self.vid_skeletons[0][i]))
        return dist



    def extract_features(self):
        video_features = []
        for vid in self.vid_skeletons:
            prev_skel = None
            features = []
            for skel in vid:
                features.append(preprocess_dataset(skel, prev_skel))
                prev_skel = skel
            video_features.append(features)
        return video_features


    def _read_skeletons(self):
        skeletons = []
        for filename, info_name in zip(os.listdir(self.label_dir), os.listdir(self.info_dir)):
            skeletons.append(self._read_file(self.label_dir + filename, self.info_dir + info_name))
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
            kp_arr = kp_file.read().strip().split("\n")
            for frame in kp_arr:
                kps = frame.split(";")
                k_arr = []
                for k in kps:
                    points = k.split(",")
                    p_arr = []
                    for p in points:
                        p_arr.append(int(float(p)))
                    k_arr.append(p_arr)
                frames.append(k_arr)
            return frames
