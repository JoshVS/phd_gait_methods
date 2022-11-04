import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

from dataset import VideoDataset
from preprocessing import dist

import mediapipe as mp
mp_pose = mp.solutions.pose
pc = mp_pose.POSE_CONNECTIONS


skel_da = VideoDataset(max_samples=20)

def show_gait_cycle(vid_seq):
    distances = []
    for i, kp in enumerate(vid_seq):
        if (kp == 0):
            continue
        # if i == 80:
        # print(kp)
    # quit()
        plt.figure()
        for p in pc:
            p1, p2 = p
            x = [kp[p1][0],kp[p2][0]]
            y = [kp[p1][1],kp[p2][1]]
            plt.plot(x,y )
        # plt.scatter(kp[:][0], kp[:][1])

        
        ankle1 = kp[mp_pose.PoseLandmark.LEFT_ANKLE]
        ankle2 = kp[mp_pose.PoseLandmark.RIGHT_ANKLE]
        x = [ankle1[0],ankle2[0]]
        y = [ankle1[1],ankle2[1]]
        plt.plot(x,y)
            

        plt.savefig(f"walk_eg/{i}.png")
        plt.close()
        
        distances.append(dist(ankle1, ankle2))
    plt.figure()
    # print(distances)
    # quit()
    # plt.plot(distances)
    # plt.plot(find_peaks(distances))
    # print(find_peaks(distances))
    # quit()
    distances = savgol_filter(distances, 9, 3)
    plt.plot(distances)
    peaks = find_peaks(distances)[0]
    peak_vals = [distances[x] for x in peaks]
    # print(peaks, peak_vals)
    # quit()
    plt.scatter(peaks, peak_vals)
    plt.savefig("output.png")
    # plt.show()
    plt.close()
    quit()

show_gait_cycle(skel_da.norm_skeletons[2])

