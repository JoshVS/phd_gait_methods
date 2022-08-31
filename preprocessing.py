from turtle import right
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

