import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset
from preprocessing import dist
from lstm_model import create_classifier

import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


skel_da = VideoDataset(max_samples=20)
X = np.array(skel_da.gait_phases)
y = skel_da.one_hot_labels()
my_classifier = create_classifier(skel_da.n_classes)
my_classifier.fit(X, y, batch_size=5, epochs=100)
# (n_samples, n_gaits, n_phases, n_features) - skel_data.gait_phases
# NOTE: There are always 7 phases and 5 features
# Therefore, it is (n_samples, n_gaits, 7, 5)
# NOTE: Final dimension needs to be flattened - put all features there, turn from 5 to whatever total features is (DONE)
# ALSO NOTE: compress (n_samples, n_gaits) into (n_samples) - each full gait cycle is a full sample