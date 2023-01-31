import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset, KinectDataset
from preprocessing import dist
from lstm_model import create_classifier

import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS

skel_data = KinectDataset(max_samples=0.025)
for i in range(10):
    skel_data.show_person(0,i)