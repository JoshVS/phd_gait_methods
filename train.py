import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset
from preprocessing import dist

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


skel_da = VideoDataset(max_samples=20)

print(skel_da.gait_cycles)