import os

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.interpolate import interp1d
from genericdataset import GenericGaitDataset

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
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


class NaiveKinectDataset(GenericGaitDataset):

    def setup_information(self):        
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

        self.upper_torso = [
            "Head",
            "Shoulder-Center",
            "Shoulder-Right",
            "Shoulder-Left"        
        ]

        self.lower_torso = [
            "Spine",
            "Hip-centro",
            "Hip-Right",
            "Hip-Left"
        ]

            
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


def get_kp_from_file(filename, kp_dict):
    cap = cv2.VideoCapture(filename)


    if not cap.isOpened():
        print(f"Error opening file {filename}")
        quit()
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = pose.process(im)
        # print(dir(lm))
        # quit()
        curr_frame = []
        if lm.pose_landmarks is not None:
            for k in kp_dict.keys():
                v = kp_dict[k]
                curr_frame.append(k)
                # print(dir(lm.pose_landmarks))
                # quit()
                curr_frame.append(lm.pose_landmarks.landmark[v].x)
                curr_frame.append(lm.pose_landmarks.landmark[v].y)

        frames.append(curr_frame)
    return frames




            


class HARDetection(GenericGaitDataset):
    def __init__(self, directory='har_detection/', max_samples=None, t_interp=6, num_dims=2, generate_test_video=None):
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video)

    def _get_file_data(self, max_samples):
        keypoints_arr = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index"
        ]
        kp_dict = {}
        for i, s in enumerate(keypoints_arr):
            kp_dict[s] = i
        subdirs = os.listdir(self.directory)
        classes = []
        videos = []
        for i, d in enumerate(subdirs):
            classes.append(d)
            print(f"Processing class {d}  [{i + 1} / {len(subdirs)}]")
            loop = tqdm(os.listdir(self.directory + d))
            for vid in loop:
                filename = os.path.join(self.directory, d, vid)
                videos.append(get_kp_from_file(filename, kp_dict))
        self.classes = classes
        return videos, kp_dict

    def setup_information(self):        
        self.step_classifier = RandomForestClassifier()        
        self.connections = [
            ("nose", "right_eye"),
            ("nose", "left_eye"),
            ("right_eye_outer", "right_eye"),
            ("left_eye_outer", "left_eye"),
            ("right_eye_inner", "right_eye"),
            ("left_eye_inner", "left_eye"),
            ("mouth_left", "mouth_right"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("left_wrist", "left_pinky"),
            ("left_wrist", "left_index"),
            ("left_wrist", "left_thumb"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("right_wrist", "right_pinky"),
            ("right_wrist", "right_thumb"),
            ("right_wrist", "right_index"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("left_ankle", "left_heel"),
            ("left_heel", "left_foot_index"),
            ("left_ankle", "left_foot_index"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("right_ankle", "right_heel"),
            ("right_heel", "right_foot_index"),
            ("right_ankle", "right_foot_index")
        ]

        self.upper_torso = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder", 
        ]

        self.lower_torso = [
            "left_hip",
            "right_hip"
        ]
