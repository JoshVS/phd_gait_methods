import os

import numpy as np
import cv2

from tqdm import tqdm

import mediapipe as mp
mp_pose = mp.solutions.pose

DATASET_DIR = "../../Datasets/MoviesGuns/"


class VideoDataset:

    def __init__(self, label_dir="labels/", info_dir="info/"):
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.skeletons = self._read_skeletons()


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
                        p_arr.append(int(p))
                    k_arr.append(p_arr)
                frames.append(k_arr)
            return frames
                    

my_dataset = VideoDataset()