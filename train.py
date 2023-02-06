import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset, KinectDataset
from preprocessing import dist
from lstm_model import create_classifier
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


callbacks = [
    EarlyStopping(patience=100),
    TensorBoard(),
    ReduceLROnPlateau()
]


# skel_data = KinectDataset(max_samples=0.0125)
skel_data = KinectDataset()

X = np.array(skel_data.gait_phases)
y = skel_data.one_hot_labels()
my_classifier = create_classifier(skel_data.n_classes, input_shape=skel_data.shape[-2:])
my_classifier.fit(X, y, batch_size=8, epochs=100, validation_split=0.15, callbacks=callbacks)