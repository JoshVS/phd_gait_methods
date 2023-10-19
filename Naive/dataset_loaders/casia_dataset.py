import os
import cv2
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .genericdataset import GenericGaitDataset


from scipy.interpolate import interp1d
# matplotlib.use('TkAgg')
np.seterr(all='raise')

from sklearn.ensemble import RandomForestClassifier

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def _dim(l, check_for_error):
    if type(l) != list and type(l) != np.ndarray:
        return []
    else:
        if type(l[0]) == list and check_for_error:
            next_dim = len(l[0])
            for mini_l in l[1:]:
                if len(mini_l) != next_dim:
                    raise ValueError("Array is sparse")
        return [len(l)] + _dim(l[0], check_for_error)

def dim(l, check_for_error=False):
    return tuple(_dim(l, check_for_error))

def read_from_cached_file(filename):
    with open(filename, 'r') as in_file:
        lines = in_file.read().split("\n")
    curr_data = []
    z_data = []
    for line in lines:
        if line == "": continue
        kp, x, y, z = line.split(";")
        curr_data.append([kp, float(x), float(y)])
        z_data.append([float(z)])
    if curr_data == []:
        return None
    return curr_data, z_data
    
        

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
                curr_frame.append([k, lm.pose_landmarks.landmark[v].x, lm.pose_landmarks.landmark[v].y, lm.pose_landmarks.landmark[v].z])
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

    def __init__(self, directory='../../../Datasets/CASIA/DatasetB-2/video/', max_samples=None, t_interp=6, num_dims=2, generate_test_video=None, extract_steps=False, test_split=0.1, val_split=0.3):
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video, extract_steps=extract_steps)
        self.val_split = val_split
        self.test_split = test_split
        self.initialise_stuff()

    def initialise_stuff(self):
        self.skel_data, self.kp_indices, self.z_data = self._get_file_data(self.max_samples) # (n_people, n_files, n_lines, 3)
        

        self.setup_information()
        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        
        self.X, self.labels = self.reshape_skeletons()
        
        
        
        self.y  = self.labels



        self.translation_vector()
        self.scaling_vector()
        # print(self.X)
        # quit()
        if self.generate_test_video is not None:
            self.show_video(self.generate_test_video)
        if self.num_dims > 2:
            self.rotation_vector(2)
            if self.generate_test_video is not None:
                self.show_video(self.generate_test_video)
        if self.generate_test_video is not None and self.generate_test_video < 0:
            quit()
   
        # quit()     
        # self.X, self.y = self.get_individual_steps()
   
        self.interpolate_by_time()
        
        self.n_classes = len(np.unique(self.y))
        self.y = self.convert_to_one_hot()
        

        # self.interpolate_by_time()
        

        # self.q = self.quality_matrices()

        self.y_raw = self.y.copy()
        # self.y = self.to_one_hot()
        # self.X = self.get_position_vectors()
        print(self.X.shape)
        print(dim(self.z_data))
        quit()
        self.split_train_and_test()
        self.gso = self.normalize_gso(self.create_graph_shift_operator())

    def convert_to_one_hot(self):
        onehot_mat = np.zeros((len(self.y), self.n_classes))
        for i in range(onehot_mat.shape[0]):
            onehot_mat[i, self.y[i]] = 1
        return onehot_mat


    def interpolate_by_time(self, convert_to_numpy=True):
        min_frames = min([len(x) for x in self.X])
        for i in range(len(self.X)):
            curr_sample = self.X[i]
            x = np.arange(len(curr_sample))
            # print(len(curr_sample), i, len(self.X))
            f = interp1d(x, curr_sample, axis=0)
            xnew = np.linspace(0, len(curr_sample) - 1, min_frames)
            # print(xnew, len(curr_sample))
            # quit()
            ynew = f(xnew)
            self.X[i] = ynew
        if convert_to_numpy:
            self.X = np.array(self.X)

    def split_train_and_test(self):
        # print(dim(self.y))
        # quit()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_split)

    def scaling_vector(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                ct, ut, lt = self.get_centroid(self.X[i][j])
                self.X[i][j] = self._scaling_vector(self.X[i][j], ut, lt)

    def _scaling_vector(self, X, ut, lt):
        diff = [a - b for (a, b) in zip(ut, lt)]
        scale_val = np.sqrt(sum([a**2 for a in diff]))
        return X / scale_val#[a / scale_val for a in X]


    def normalize_gso(self, gso):
        Ident_mat = np.identity(gso.shape[1], dtype=np.float32)
        In = self._norm(gso)
        Out = self._norm(gso[::-1, ::-1])
        return np.stack((Ident_mat, In, Out))


    
    def _norm(self, X):
        node_degrees = np.sum(X, 0)
        w = X.shape[1]
        degree_to_normalize = np.zeros((w, w))
        for i in range(w):
            if node_degrees[i] > 0:
                degree_to_normalize[i,i] = node_degrees[i]**-1
        return np.dot(X, degree_to_normalize)


    def create_graph_shift_operator(self):
        gso = np.zeros((self.X.shape[-2], self.X.shape[-2]))
        for i in range(len(self.edge_matrix[0])):
            ind1 = self.edge_matrix[0][i]
            ind2 = self.edge_matrix[1][i]
            gso[ind1, ind2] = 1
        # gso = np.empty((len(self.edge_matrix[0]), len(self.edge_matrix)))
        # for i in range(gso.shape[0]):
        #     gso[i, 0] = self.edge_matrix[0][i]
        #     gso[i, 1] = self.edge_matrix[1][i]
        return gso
        
    def get_centroid(self, X):
        # Step 1: Get upper Centroid and Lower Centroid
        upper_limbs = []
        for u in self.upper_torso:
            upper_limbs.append(X[self.kp_indices[u]])
        # uc = [sum(upper_limbs[x::self.num_dims]) / len(self.upper_torso) for x in range(self.num_dims)]
        uc = np.einsum("ij-> j", upper_limbs)
      


        lower_limbs = []
        for l in self.lower_torso:
            lower_limbs.append(X[self.kp_indices[l]])
        lc = np.einsum("ij-> j", lower_limbs)

        return np.array([(x + y) / 2 for (x, y) in zip(uc, lc)]), uc, lc

    def rotation_vector(self, tm):
        print("Creating Rotation Matrix")
        loop = tqdm(range(len(self.X)))
        for i in loop:
            
            centroids = []
            for j in range(len(self.X[i])):                
                centroids.append( self.get_centroid(self.X[i][j]))

                ct, ut, lt = centroids[-1]
                if j > tm:
                    ang = 0
                    ct_prev, _,_ = centroids[j - tm]
                    self.X[i][j], ang = self._rotation_vector(self.X[i][j], self.X[i][j - tm], ct, ut, lt, ct_prev, ang)

    def _rotation_vector(self, X, Xp, ct, ut, lt, ct_prev, def_ang):
        d = np.sqrt(sum([(a - b)**2 for a, b in zip(ct, ct_prev)]))
        # try:
        # rmov = []
        # for a, b in zip(ct, ct_prev):
        #     try:
        #         rmov.append((a - b) / d)
        #     except FloatingPointError:
        #         print(a, b, (a - b), d)
        #         quit()
        if d > 0:
            rmov = [(a - b) / d for a, b in zip(ct, ct_prev)]
            # except FloatingPointError:
                # print(a, b, (a - b),)
            self.rmov = rmov
            x = np.abs(rmov[0] - ct[0])
            r = np.sqrt((rmov[0] - ct[0]) ** 2 + (rmov[2] - ct[2]) ** 2)
            ang = np.arcsin(x / r) if r > 0 else 0
        else:
            ang = def_ang

        R_inv = [
            [np.cos(ang), 0, np.sin(ang)],
            [0, 1, 0],
            [-np.sin(ang), 0, np.cos(ang)]
        ]

        # dcen = np.sqrt(sum([(a - b)**2 for a, b in zip(ut, lt)]))
        # rtop = [(a - b) / dcen for a, b in zip(ut, lt)]
        # self.rtop = rtop

        # cross_prod = np.cross(rtop, rmov)
        # dcross = np.sqrt(sum(a**2 for a in cross_prod))
        # rleft = [a / dcross for a in cross_prod]
        # self.rleft = rleft

        # R_mat = np.array([
        #     rmov,
        #     rtop,
        #     rleft
        # ])

        # R_inv = R_mat # np.linalg.inv(R_mat)

        # print(R_inv.shape)
        # quit()

        ret_val = []
        for i in range(len(X) // self.num_dims):
            curr_vals = X[i * self.num_dims: i * self.num_dims + self.num_dims]
            # print(np.dot(R_inv, curr_vals).shape)
            # quit()
            ret_val += list(np.dot(R_inv, curr_vals))

        return ret_val, ang


        # cross_prod = np.cross(rtop, rmov)
        

    def translation_vector(self):
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                ct, ut, lt = self.get_centroid(self.X[i][j])
                self.X[i][j] = self._translation_vector(self.X[i][j], ct)

    def _translation_vector(self, X, ct):        
        # Step 2: Translation Vector
        pt = ct.T
        return X - pt

    def reshape_skeletons(self):
        # Need skeleton to be shape (vid_seq, frame, keypoints, 2)
        labels = []
        reshaped_skel = []
        reshaped_z_data = []
        for i, person in enumerate(self.skel_data):
            # reshaped_skel.append([])
            for f in person:
                reshaped_skel.append([])
                labels.append(i)
                for l in f:
                    if l[0] == self.headpoint:
                        reshaped_skel[-1].append([])
                    reshaped_skel[-1][-1].append(l[1:])
        return reshaped_skel, labels

    def create_file_data(self, kp_dict, max_samples):
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
        z_data = []
        vid_filenames = []
        person_id = 1
        def create_person_string(p_id):
            return f"{'0' if p_id < 10 else ''}{p_id}"
        p_string = create_person_string(person_id)
        list_of_files = glob.glob(f"*-*-{p_string}-*.avi", root_dir=self.directory)
        if max_samples is None:
            pass
        elif 0 < max_samples < 1:
            list_of_files = list_of_files[:int(len(list_of_files) * max_samples)]
        elif max_samples < len(list_of_files):
            list_of_files = list_of_files[:int(max_samples)]
            
        while len(list_of_files) > 0:
            
            if max_samples is None:
                pass
            elif 0 < max_samples < 1:
                list_of_files = list_of_files[:int(len(list_of_files) * max_samples)]
            elif max_samples < len(list_of_files):
                list_of_files = list_of_files[:int(max_samples)]
            classes.append(person_id)
            class_vids = []
            z_class_vids = []
            class_vid_filenames = []

            loop = tqdm(list_of_files)
            if not os.path.exists(f"cached/{p_string}/"):
                os.makedirs(f"cached/{p_string}/")

            for i, vid in enumerate(loop):
                if os.path.exists(f"cached/{p_string}/{vid}.txt"):
                    to_append, z_data = read_from_cached_file(f"cached/{p_string}/{vid}.txt")
                    if to_append is not None:
                        class_vids.append(to_append)
                        z_class_vids.append(z_data)
                        vid_filenames.append(os.path.join(self.directory, vid))
                else:
                    print(f"File cached/{p_string}/{vid}.txt doesn't exist, creating")
                    filename = os.path.join(self.directory, vid)
                    c = get_kp_from_file(filename, kp_dict)
                    class_vids.append(c)
                    vid_filenames.append(filename)
                    write_to_file(f"cached/{p_string}/{vid}.txt", c)
            videos.append(class_vids)
            z_data.append(z_class_vids)


            person_id += 1
            p_string = create_person_string(person_id)
            list_of_files = glob.glob(f"*-*-{p_string}-*.avi", root_dir=self.directory)

        self.video_filenames = vid_filenames
        self.classes = classes
        return videos, kp_dict, z_data
            

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
        return self.create_file_data(kp_dict, max_samples)
        

    def setup_information(self):        
        self.step_classifier = RandomForestClassifier()        
        self.connections = [
            ('nose', 'right_eye_inner'),
            ('nose', 'left_eye_inner'),
            ('left_eye_inner', 'left_eye'),
            ('left_eye', 'left_eye_outer'),
            ('left_eye_outer', 'left_ear'),
            
            ('right_eye_inner', 'right_eye'),
            ('right_eye', 'right_eye_outer'),
            ('right_eye_outer', 'right_ear'),

            ('mouth_left', 'mouth_right'),

            ('left_shoulder', 'right_shoulder'),

            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_thumb'),
            ('left_wrist', 'left_index'),
            ('left_wrist', 'left_pinky'),

            
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_thumb'),
            ('right_wrist', 'right_index'),
            ('right_wrist', 'right_pinky'),

            ('right_shoulder', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_hip', 'left_hip'),

            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot_index'),

            
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot_index'),

        ]

        self.edge_matrix = [[self.kp_indices[x] for x, _ in self.connections], [self.kp_indices[y] for _, y in self.connections]]

        self.upper_torso = [
            "nose",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "right_ear",
            
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "left_ear",

            "left_shoulder",
            "right_shoulder"
        ]

        self.lower_torso = [
            "left_hip",
            "right_hip"
        ]
        self.headpoint = "nose"
        self.left_elbow = "left_elbow"
        self.right_elbow = "right_elbow"
        self.left_knee = "left_knee"
        self.right_knee = "right_knee"
        self.right_wrist = "right_wrist"
        self.left_wrist = "left_wrist"
        self.left_ankle = "left_ankle"
        self.right_ankle = "right_ankle"

    def create_edge_matrix(self):
        self.split_train_and_test()
        

        