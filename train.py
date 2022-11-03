import matplotlib.pyplot as plt

from dataset import VideoDataset
from preprocessing import dist

import mediapipe as mp
mp_pose = mp.solutions.pose


skel_da = VideoDataset()

def show_gait_cycle(vid_seq):
    distances = []
    for kp in vid_seq:
        
        ankle1 = kp[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle2 = kp[mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        distances.append(dist(ankle1, ankle2))
    plt.figure()
    plt.plot(distances)
    plt.savefig("output.png")
    plt.close()

show_gait_cycle(skel_da.vid_skeletons[0])

