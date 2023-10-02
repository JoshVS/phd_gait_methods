import os
import cv2
import glob

import numpy as np
from tqdm import tqdm
from .genericdataset import GenericGaitDataset


# matplotlib.use('TkAgg')
np.seterr(all='raise')

from sklearn.ensemble import RandomForestClassifier

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def read_from_cached_file(filename):
    with open(filename, 'r') as in_file:
        lines = in_file.read().split("\n")
    curr_data = []
    for line in lines:
        if line == "": continue
        kp, x, y = line.split(";")
        curr_data.append([kp, float(x), float(y)])
    if curr_data == []:
        return None
    return curr_data
    
        

def get_kp_from_file(filename, kp_dict):
    cap = cv2.VideoCapture(filename)


    if not cap.isOpened():
        print(f"Error opening file {filename}")
        quit()
    frames = []
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loop = tqdm(range(length))
    for i in loop:
        ret, frame = cap.read()
        if not ret:
            break
        
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = pose.process(im)
        # print(dir(lm))
        # quit()
        curr_frame = []
        if lm.pose_landmarks is not None:
            for k in kp_dict.keys():
                v = kp_dict[k]
                curr_frame.append([k, lm.pose_landmarks.landmark[v].x, lm.pose_landmarks.landmark[v].y])
                # print(dir(lm.pose_landmarks))
                # quit()

        frames.append(curr_frame)
    return frames




def write_to_file(filename, kps):
    lines = []
    for kp in kps:
        curr_line = []
        for p in kp:
            c = []
            for i in p:
                c.append(str(i))
            curr_line.append(";".join(c))
        lines.extend(curr_line)
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines))

class CASIADataset(GenericGaitDataset):

    def __init__(self, directory='../../../Datasets/CASIA/DatasetB-2/video/', max_samples=None, t_interp=6, num_dims=2, generate_test_video=None, extract_steps=False):
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video, extract_steps=extract_steps)
        self.initialise_stuff()

    def create_file_data(self, kp_dict):
        if not os.path.exists("cached/"):
            os.makedirs("cached")
        # subdirs = os.listdir(self.directory)
        # classes = []
        # videos = []
        # for i, d in enumerate(subdirs):
        #     classes.append(d)
        #     print(f"Processing class {d}  [{i + 1} / {len(subdirs)}]")
        #     loop = os.listdir(self.directory + d)
        #     class_vids = []
        #     if not os.path.exists(f"cached/{d}/"):
        #         os.makedirs(f"cached/{d}/")
        #     loop = tqdm(loop)
        #     for i, vid in enumerate(loop):
        #         if os.path.exists(f"cached/{d}/{vid}.txt"):
        #             to_append = read_from_cached_file(f"cached/{d}/{vid}.txt")
        #             if to_append is not None:
        #                 class_vids.append(to_append)
        #         else:
        #             print(f"File cached/{d}/{vid}.txt doesn't exist, creating")
        #             filename = os.path.join(self.directory, d, vid)
        #             c = get_kp_from_file(filename, kp_dict)
        #             class_vids.append(c)
        #             write_to_file(f"cached/{d}/{vid}.txt", c)
        #     videos.append(class_vids)
        # self.classes = classes
        # return videos, kp_dict
        classes = []
        videos = []
        person_id = 1
        def create_person_string(p_id):
            return f"{'0' if p_id < 10 else ''}{p_id}"
        p_string = create_person_string(person_id)
        list_of_files = glob.glob(f"*-*-{p_string}-*.avi", root_dir=self.directory)
        while len(list_of_files) > 0:
            classes.append(person_id)
            class_vids = []

            loop = tqdm(list_of_files)
            if not os.path.exists(f"cached/{p_string}/"):
                os.makedirs(f"cached/{p_string}/")

            for i, vid in enumerate(loop):
                if os.path.exists(f"cached/{p_string}/{vid}.txt"):
                    to_append = read_from_cached_file(f"cached/{p_string}/{vid}.txt")
                    if to_append is not None:
                        class_vids.append(to_append)
                else:
                    print(f"File cached/{p_string}/{vid}.txt doesn't exist, creating")
                    filename = os.path.join(self.directory, vid)
                    c = get_kp_from_file(filename, kp_dict)
                    class_vids.append(c)
                    write_to_file(f"cached/{p_string}/{vid}.txt", c)
            videos.append(class_vids)


            person_id += 1
            p_string = create_person_string(person_id)
            list_of_files = glob.glob(f"*-*-{p_string}-*.avi", root_dir=self.directory)
            
        self.classes = classes
        return videos, kp_dict
            

    def _get_file_data(self, max_samples):
        keypoints_arr = [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index"
        ]
        kp_dict = {}
        for i, s in enumerate(keypoints_arr):
            kp_dict[s] = i
        return self.create_file_data(kp_dict)
        

    def setup_information(self):        
        self.step_classifier = RandomForestClassifier()        
        self.connections = [
            ('Head', 'Shoulder-Center'),
            ('Shoulder-Center', 'Shoulder-Right'),
            ('Shoulder-Center', 'Shoulder-Left'),
            ('Shoulder-Center', 'Spine'),
            ('Spine', 'Hip-centro'),
            ('Hip-centro', 'Hip-Left'),            
            ('Hip-centro', 'Hip-Right'),            
            ('Hip-Right', 'Knee-Right'),            
            ('Hip-Left', 'Knee-Left'),            
            ('Knee-Right', "Ankle-Right"),                    
            ("Ankle-Right", 'Foot-Right'),        
            ('Knee-Left', "Ankle-Left"),                    
            ("Ankle-Left", 'Foot-Left'),
            ("Shoulder-Left", "Elbow-Left"),
            ("Elbow-Left", "Wrist-Left"),
            ("Wrist-Left", "Hand-Left"),
            ("Shoulder-Right", "Elbow-Right"),
            ("Elbow-Right", "Wrist-Right"),
            ("Wrist-Right", "Hand-Right"),
        ]

        self.upper_torso = [
            "Head",
            "Shoulder-Center",
            "Shoulder-Right",
            "Shoulder-Left"        
        ]

        self.lower_torso = [
            "Spine",
            "Hip-centro",
            "Hip-Right",
            "Hip-Left"
        ]
        self.headpoint = "Head"
        self.left_elbow = "Elbow-Left"
        self.right_elbow = "Elbow-Right"
        self.left_knee = "Knee-Left"
        self.right_knee = "Knee-Right"
        self.right_wrist = "Wrist-Right"
        self.left_wrist = "Wrist-Left"
        self.left_ankle = "Ankle-Left"
        self.right_ankle = "Ankle-Right"

            