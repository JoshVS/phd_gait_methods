import numpy as np
import cv2

from tqdm import tqdm

import mediapipe as mp
mp_pose = mp.solutions.pose

def extract_keypoints(image):
    my_pose = mp_pose.Pose(
            static_image_mode = True,
            model_complexity = 2,
            enable_segmentation = True,
            min_detection_confidence = 0.5
        )
    image_height, image_width,_ = image.shape
    with my_pose as pose:
        keypoints = pose.process(image)
        if keypoints.pose_landmarks is not None:
            for l in keypoints.pose_landmarks.landmark:
                l.x *= image_width
                l.y *= image_height
    return keypoints



def extract_from_video(filename, outfile_name):
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loop = tqdm(range(length))

    kp_arr = []
    with open(outfile_name, "w") as out_file:
        for i in loop:
            curr_kp = []
            _, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            kp = extract_keypoints(image)
            if kp.pose_landmarks is not None:
                for l in kp.pose_landmarks.landmark:
                    curr_kp.append((l.x, l.y))
            else:
                curr_kp.append((-1, -1))
            
            kp_arr.append(curr_kp)
            out_file.write(f"{curr_kp}\n")
            
            # cv2.imshow('frame',image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            loop.set_postfix()

    cap.release()

extract_from_video("1.mp4", "kp_results.txt")