import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose


pc = mp_pose.POSE_CONNECTIONS

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def global_coordinate_frame(kp_arr):
    left_hip = kp_arr.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = kp_arr.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    center_hip = ((left_hip.x + right_hip.x) / 2, (left_hip.x + right_hip.x) / 2)

    for l in kp_arr.pose_landmarks.landmark:
        l.x -= center_hip[0]
        l.y -= center_hip[1]
    
    return kp_arr

def dist(point1, point2):
    term1 = (point1.x + point2.x) ** 2
    term2 = (point1.y + point2.y) ** 2
    return np.sqrt(term1 + term2)

def keypoints_to_unit_vectors(kp_arr):
    unit_vectors = []
    for p in pc:
        p1_ind, p2_ind = p
        point1 = kp_arr.pose_landmarks.landmark[p1_ind]
        point2 = kp_arr.pose_landmarks.landmark[p2_ind]
        point_dist = dist(point1, point2)
        unit_vectors.append(Vector(
            (point2.x - point1.x) / point_dist,
            (point2.y - point1.y) / point_dist
        ))
    return unit_vectors


preprocess_dataset = lambda x: keypoints_to_unit_vectors(global_coordinate_frame(x))
