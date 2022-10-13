import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose

def global_coordinate_frame(kp_arr):
    left_hip = kp_arr.pose_landmarks.landmark[mp_pose.LEFT_HIP]
    right_hip = kp_arr.pose_landmarks.landmark[mp_pose.RIGHT_HIP]

    center_hip = ((left_hip.x + right_hip.x) / 2, (left_hip.x + right_hip.x) / 2)

    for l in kp_arr.pose_landmarks:
        l.x -= center_hip[0]
        l.y -= center_hip[1]
    
    return kp_arr

def dist(point1, point2):
    term1 = (point1.x + point2.x) ** 2
    term2 = (point1.y + point2.y) ** 2
    return np.sqrt(term1 + term2)

def keypoints_to_unit_vectors(kp_arr, pc_arr):
    unit_vectors = []
    for pc in pc_arr:
        point1 = kp_arr[pc[0]]
        point2 = kp_arr[pc[1]]
        dist = dist(point1, point2)
        unit_vectors.append([
            (point2.x - point1.x) / dist,
            (point2.y - point1.y) / dist
        ])
    return unit_vectors
