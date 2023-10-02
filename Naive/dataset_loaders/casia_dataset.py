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

class CASIADataset(GenericGaitDataset):

    def _get_file_data(self, max_samples):
        

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
        self.headpoint = "Head"
        self.left_elbow = "Elbow-Left"
        self.right_elbow = "Elbow-Right"
        self.left_knee = "Knee-Left"
        self.right_knee = "Knee-Right"
        self.right_wrist = "Wrist-Right"
        self.left_wrist = "Wrist-Left"
        self.left_ankle = "Ankle-Left"
        self.right_ankle = "Ankle-Right"

            