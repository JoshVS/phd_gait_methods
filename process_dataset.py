import os

import numpy as np
import cv2

from tqdm import tqdm

from preprocessing import preprocess_dataset

import mediapipe as mp
mp_pose = mp.solutions.pose

DATASET_DIR = "../../Datasets/CASIA/DatasetB-1/video/"



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



def extract_from_video(filename, outfile_name, infofile_name):
    # Open up video file, get number of frames
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loop = tqdm(range(length))
    
    kp_arr = [] # This is an inline comment

    # Output processed data to file
    with open(outfile_name, "w")  as out_file :
        dimensions = None
        for i in loop:

            # Create KP array for current frame
            curr_kp = []

            # Read in frame, convert to RGB colouring
            _, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Write metadata to info file
            if i == 0:
                with open(infofile_name, "w") as info_file:
                    height, width, _ = image.shape
                    info_file.write(f"height, {height}; width, {width}")

            # Extract keypoints
            kp = extract_keypoints(image)
            
            # Add keypoints to array
            if kp.pose_landmarks is not None:
                for l in kp.pose_landmarks.landmark:
                    curr_kp.append((l.x, l.y))
            else:
                for _ in range(33):
                    curr_kp.append((-1, -1))
                         
            # Format keypoints, write to output file
            kp_write = [str(x) + "," + str(y) for x, y in curr_kp]
            out_file.write(f"{';'.join(kp_write)}\n")
            
            loop.set_postfix()

    cap.release()

def extract_from_directory(d_dir=DATASET_DIR):
    num_files = len(os.listdir(d_dir))
    for i, filename in enumerate(os.listdir(d_dir)):
        if filename.split(".")[-1] != "mp4" and filename.split(".")[-1] != "avi":
            continue
        
        print(f"EXTRACTING {d_dir + filename} [{i + 1} / {num_files}]")
        if filename.split(".")[-1] == "avi":
            extract_from_video(
                d_dir + filename, 
                "labels/" + filename + ".txt",
                "info/" + filename + ".info")
        



extract_from_directory()