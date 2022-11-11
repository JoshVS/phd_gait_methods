import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose


pc = mp_pose.POSE_CONNECTIONS

class Vector:
    def __init__(self, x, y):
        self[0] = x
        self[1] = y

    

def global_coordinate_frame(kp_arr):
    # print(kp_arr)
    # quit()
    left_hip = kp_arr[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = kp_arr[mp_pose.PoseLandmark.RIGHT_HIP]

    center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

    for l in kp_arr:
        l[0] -= center_hip[0]
        l[1] -= center_hip[1]
    
    return kp_arr

def dist(point1, point2):
    term1 = (point1[0] - point2[0]) ** 2
    term2 = (point1[1] - point2[1]) ** 2
    return np.sqrt(term1 + term2)

def keypoints_to_unit_vectors(kp_arr):
    unit_vectors = []
    for p in pc:
        p1_ind, p2_ind = p
        point1 = kp_arr[p1_ind]
        point2 = kp_arr[p2_ind]
        dx = point2[0] - point1[0]
        point_dist = dist(point1, point2)
        # unit_vectors.append(Vector(
        #     (point2[0] - point1[0]) / point_dist,
        #     (point2[1] - point1[1]) / point_dist
        # ))
        unit_vectors.append(0 if point_dist == 0 else   np.arccos(dx / point_dist))
    return unit_vectors


def body_part_feature_lengths(kp_arr):
    """
    Extracts the distance of each joint
    This shouldn't vary too much in dataset
    """
    lengths = []
    for p in pc:
        p1_ind, p2_ind = p
        point1 = kp_arr[p1_ind]
        point2 = kp_arr[p2_ind]
        point_dist = dist(point1, point2)
        lengths.append(point_dist)
    return lengths

def join_distance_features(kp_arr):
    joint_distances = []
    for i, j in np.ndindex((len(kp_arr), len(kp_arr))):
        if i >= j:
            continue
        point1 = kp_arr[i]
        point2 = kp_arr[j]
        point_dist = dist(point1, point2)
        joint_distances.append(point_dist)
    return joint_distances
    
def inter_frame_distances(kp_arr1, kp_arr2):
    if kp_arr2 is None:
        ret_zeros = []
        for i in range(len(kp_arr1)):
            ret_zeros.append(0)
        return ret_zeros
    if_distances = []
    for i in range(len(kp_arr1)):
        point1 = kp_arr1[i]
        point2 = kp_arr2[i]
        if_distances.append(dist(point1, point2))
    return if_distances


def inter_frame_angles(kp_arr1, kp_arr2):
    if kp_arr2 is None:
        ret_zeros = []
        for i in range(len(kp_arr1)):
            ret_zeros.append(0)
        return ret_zeros
    angles1 = keypoints_to_unit_vectors(kp_arr1)
    angles2 = keypoints_to_unit_vectors(kp_arr2)
    d_angles = []
    for i in range(len(angles1)):
        d_angles.append(angles2[i] - angles1[i])

    return d_angles



def preprocess_dataset(x, y):
    glob_coords_x = x#global_coordinate_frame(x)
    glob_coords_y = y#y if y is None else global_coordinate_frame(y)
    return [
        keypoints_to_unit_vectors(glob_coords_x),
        body_part_feature_lengths(glob_coords_x),
        join_distance_features(glob_coords_x),
        inter_frame_distances(glob_coords_x, glob_coords_y),
        inter_frame_angles(glob_coords_x, glob_coords_y)
    ]
