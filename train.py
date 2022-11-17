import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset
from preprocessing import dist

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


skel_da = VideoDataset(max_samples=20)

# (n_samples, n_gaits, n_phases, n_features) - skel_data.gait_phases
# NOTE: There are always 7 phases and 5 features
# Therefore, it is (n_samples, n_gaits, 7, 5)
# NOTE: Final dimension needs to be flattened - put all features there, turn from 5 to whatever total features is (DONE)
# ALSO NOTE: compress (n_samples, n_gaits) into (n_samples) - each full gait cycle is a full sample