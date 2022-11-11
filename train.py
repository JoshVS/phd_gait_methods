import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset
from preprocessing import dist

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


skel_da = VideoDataset(max_samples=20)

def filter_peaks(peaks, threshold=5):
    prev_peak = 0
    new_peaks = []
    for p in peaks:
        if p - prev_peak > threshold:
            new_peaks.append(p)
        prev_peak = p
    return new_peaks

def get_peaks(norm_skel_vid):
    distances = []
    for i, kp in enumerate(norm_skel_vid):
        ankle1 = kp[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle2 = kp[mp_pose.PoseLandmark.RIGHT_ANKLE]
        distances.append(dist(ankle1, ankle2))
    filtered_distances = savgol_filter(distances, 9, 3)
    peaks = find_peaks(filtered_distances)[0]
    peaks = filter_peaks(peaks)
    return peaks, [filtered_distances[x] for x in peaks], filtered_distances

def get_gait_cycles(norm_skel_vid):
    """
    Gait cycles happen every second time ankle distances peak
    """
    peaks, _, _ = get_peaks(norm_skel_vid)
    return peaks[::2]

def show_gait_cycle(vid_seq):
    peaks, peak_vals, distances = get_peaks(vid_seq)
    plt.plot(distances)
    plt.scatter(peaks, peak_vals)
    plt.savefig("output.png")
    plt.close()
    quit()

show_gait_cycle(skel_da.norm_skeletons[0])

