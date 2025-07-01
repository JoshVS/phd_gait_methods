import os
import cv2

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .genericdataset import GenericGaitDataset
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
np.seterr(all='raise')

from celluloid import Camera

torch.set_default_dtype(torch.float32)
yolo_model = YOLO("yolov8x-pose-p6.pt")

READ_FROM_CACHE = True
WRITE_TO_CACHE = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Thanks Gemini
def make_keypoints_rotation_invariant(keypoints, shoulder_indices=(5, 6), reference_point_index=0):
    """
    Makes a set of 2D keypoints rotation-invariant.

    This function normalizes the rotation of a pose by aligning a reference bone
    (e.g., shoulders) with the horizontal axis. It then translates the pose
    so a specific reference point (e.g., nose) is at the origin.

    Args:
        keypoints (list or np.array): A Python list of lists or a NumPy array of shape (N, 2)
                                      where N is the number of keypoints, and each inner list/row
                                      is an (x, y) coordinate. Ultralytics models typically output
                                      (x, y, confidence) so ensure you pass only the (x, y) part.
        shoulder_indices (tuple): A tuple (left_shoulder_index, right_shoulder_index)
                                  representing the indices of the shoulder keypoints
                                  in the `keypoints` array. Default assumes COCO format
                                  where 5 is left shoulder and 6 is right shoulder.
        reference_point_index (int): The index of the keypoint to use as the
                                     translation reference (e.g., nose). This
                                     point will be moved to (0,0) after rotation.
                                     Default assumes COCO format where 0 is nose.

    Returns:
        np.array: A NumPy array of shape (N, 2) with rotation-invariant keypoints.
                  The keypoints are rotated and translated.
    """
    # Convert input list to NumPy array if it's not already one
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints, dtype=float)

    if keypoints.shape[0] < max(shoulder_indices) + 1:
        print("Warning: Not enough keypoints for specified shoulder indices.")
        # Return original keypoints if reference points are missing
        return keypoints

    # Extract shoulder keypoints
    left_shoulder = keypoints[shoulder_indices[0]]
    right_shoulder = keypoints[shoulder_indices[1]]

    # Calculate the vector from left to right shoulder
    shoulder_vector = right_shoulder - left_shoulder

    # Calculate the angle of the shoulder vector with the positive x-axis
    # atan2 gives the angle in radians, handling all quadrants
    angle_rad = np.arctan2(shoulder_vector[1], shoulder_vector[0])

    # Create a 2D rotation matrix for rotation by -angle_rad
    # This rotates the pose so the shoulder line becomes horizontal
    cos_val = np.cos(-angle_rad)
    sin_val = np.sin(-angle_rad)
    rotation_matrix = np.array([
        [cos_val, -sin_val],
        [sin_val, cos_val]
    ])

    # Translate all keypoints so the left shoulder is at the origin before rotation
    # This ensures rotation happens around a point related to the pose itself
    translated_keypoints = keypoints - left_shoulder

    # Apply the rotation to all keypoints
    rotated_keypoints = np.dot(translated_keypoints, rotation_matrix.T)

    # Now, translate the rotated keypoints so the chosen reference point (e.g., nose)
    # is at the origin (0,0). This makes the pose translation-invariant.
    translation_vector = rotated_keypoints[reference_point_index]
    rotation_invariant_keypoints = rotated_keypoints - translation_vector

    return rotation_invariant_keypoints

def dist(a, b):
    return np.sqrt(sum([(x - y) ** 2 for (x, y) in zip(a, b)]))

def centroid(a, b):
    return [(x + y) / 2 for (x, y) in zip(a, b)]

def reframe_coords(coords, center_point, norm_dist):
    if norm_dist == 0:
        norm_dist = 1
    new_coords = []
    for c in coords:
        new_coords.append([(c[0] - center_point[0]) / norm_dist, (c[1] - center_point[1]) / norm_dist])
    return np.array(new_coords, dtype=np.float32)

def normalize_skeleton(skeleton):
    """
    Normalizes skeleton

    This function normalizes the rotation of a pose by aligning a reference bone
    (e.g., shoulders) with the horizontal axis. It then translates the pose
    so a specific reference point (e.g., nose) is at the origin.

    Args:
        keypoints (list or np.array): A Python list of lists or a NumPy array of shape (N, 2)
                                      where N is the number of keypoints, and each inner list/row
                                      is an (x, y) coordinate. Ultralytics models typically output
                                      (x, y, confidence) so ensure you pass only the (x, y) part.

    Returns:
        np.array: A NumPy array of shape (17, 2) with rotation-invariant keypoints.
                  The keypoints are rotated and translated.
    """


    return make_keypoints_rotation_invariant(skeleton, shoulder_indices=(5, 6), reference_point_index=0)





def cell_callback_factory(num_frames):

    def cell_callback(curr_frame, total_frames):
        # if curr_frame % (num_frames // 10) == 0:
        print(f"Frame {curr_frame} / {num_frames}")
    return cell_callback

def _dim(l, check_for_error):
    if type(l) != list and type(l) != np.ndarray:
        return []
    else:
        if len(l) == 0:
            return [0]
        if type(l[0]) == list and check_for_error:
            next_dim = len(l[0])
            for mini_l in l[1:]:
                if len(mini_l) != next_dim:
                    raise ValueError("Array is sparse")
        return [len(l)] + _dim(l[0], check_for_error)

def dim(l, check_for_error=False):
    return tuple(_dim(l, check_for_error))

def read_from_cached_file(filename, min_samples=2):
    """
    Reads in a cached keypoint file and extracts keypoints in that file.

    Args:
        filename : The filename of the keypoints. 

    Returns:
        list :  shape (timesteps, 17, 4). Each timestep contains 17 keypoints, and each keypoint contains i (person index), k (keypoint name), x and y (normalized to the bounding box).
    """
    with open(filename, 'r') as in_file:
        lines = in_file.read().split("\n")
    curr_data = []
    for line in lines:
        if line == "": continue
        i, kp, x, y = line.split(";")
        curr_data.append([int(i), kp, float(x), float(y)])
    if curr_data == [] or (len(curr_data) // 33) * 2 < min_samples:
        return None
    return curr_data
    


def get_kp_from_video(filename, kp_dict, min_frames = 2, im_height=256, im_width=256):    
    """
    Reads in a video file and extracts keypoints using a YOLOv8 pose model.

    Args:
        filename : The filename of the video to be processed. 
        kp_dict: A dictionary mapping keypoint names to their indices. There should be 17 keypoints, and kp_dict[k] = index of keypoint k.

    Returns:
        list :  list is of shape (timesteps, 17, 4). Each timestep contains 17 keypoints, and each 
                keypoint contains i, the person index, k, the keypoint (str) and x and y, normalized to the bounding box.
    """
    
    try:
        cv_frames = cv2.VideoCapture(filename)
        if not cv_frames.isOpened():
            return None
        frames = []
        ret, frame = cv_frames.read()
        while ret:
            frames.append(frame)
            ret, frame = cv_frames.read()
    except Exception as e:
        return None
    # bounding_boxes = yolo_model([os.path.join(filename, f) for f in frames], save=False, verbose=False)
    

    frame_results = []

    for  frame in frames:
        curr_frame = []

      
        found_frame = False
        # print(os.path.join(filename, frame))
        # quit()
        try:
            results_generator = yolo_model(source=frame, show=False, conf=0.3, save=False, stream=True, verbose=False)
            # print(len(results_generator))
            # quit()
        except Exception as e:
            continue
        for i in range(2):
            for k in kp_dict.keys():
                curr_frame.append([i, k, 0, 0])
        
        for i, res in enumerate(results_generator):
            if i >= 2:
                break
            
            kpts = res.keypoints.xy.cpu().numpy()[0, ...] # Shape (17, 2)
            if kpts.shape[0] != 0:
                found_frame = True  
                
                bbox = res.boxes.xywh.cpu().numpy()[0,...]
                xA, yA, w, h = bbox
                for k in kp_dict.keys():
                    v = kp_dict[k]
                    curr_frame[kpts.shape[0] * i + v] = [i, 
                                                         k, 
                                                         (kpts[v, 0] - xA) / w, 
                                                         (yA - kpts[v, 1]) / h]
                    
        if found_frame:
            frame_results.extend(curr_frame)
    if len(frame_results) < (min_frames*len(kp_dict.keys()) * 2):
        print(f"{filename} has less than {min_frames} frames, skipping")
        return None
    return frame_results


def write_to_file(filename, kps):
    """
    Writes keypoints to a cached file

    Args:
        filename : The filename to be written to.
        kps : A list of shape (timesteps, 17, 4). Each timestep contains 17 keypoints, and each keypoint contains i (person index), k (keypoint name), x and y (normalized to the bounding box).

    Returns:
        Nothing
    """
    
    lines = []
    if kps == "":
        with open(filename, "w") as outfile:
            outfile.write("")
        return
    for kp in kps:
        curr_line = [";".join([str(x) for x in kp])]
        lines.extend(curr_line)
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines))

class ShopLiftingDataset(GenericGaitDataset):

    def __init__(self, directory='../../../Datasets/ShopliftingDataset/Dataset/', max_samples=None, min_samples=5, t_interp=6, num_dims=2, generate_test_video=None, extract_steps=False, test_split=0.1, val_split=0.3, max_classes=None, num_timesteps=12, exclude_classes=None):
        
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video, extract_steps=extract_steps)
        self.val_split = val_split
        self.min_samples = min_samples
        self.exclude_classes = exclude_classes
        self.test_split = test_split
        self.max_classes = max_classes
        self.num_timesteps = num_timesteps
        self.initialise_stuff()

    def initialise_stuff(self):
        torch.set_default_dtype(torch.float32)
        self.skel_data, self.kp_indices, self.labels = self._get_file_data(self.max_samples, self.max_classes) # (n_people, n_files, n_lines, 3)
        # print(self.skel_data)
        # quit()
        
        
        
        
        self.setup_information()

        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        
        print("RESHAPING AND NORMALIZING SKELETONS")
        self.X = self.reshape_skeletons()
        self.X = self.n_skel()
        # print(dim(self.X))
        # quit()
        
        
        self.y  = self.labels

        # print(self.X[1][0])
        # quit()
        if self.generate_test_video is not None:
            self.show_video(self.generate_test_video)
   

        # self.translation_vector()
        # self.scaling_vector()
        # quit()     
        print("INTERPOLATING BY TIME")

        self.interpolate_by_time()
        self.stratify_y = self.y.copy()
        # print(self.y)
        # quit()
        # sns.histplot(np.array(self.y), x=self.classes)
        y_unique, counts = np.unique(self.y, return_counts=True)
        plt.figure()
        plt.bar(self.classes, counts)
        plt.xticks(rotation=90)
        plt.savefig("class_dist.png")
        plt.close()


        
        self.n_classes = len(np.unique(self.y))
        self.y = self.convert_to_one_hot()
        

        # self.interpolate_by_time()
        

        # self.q = self.quality_matrices()

        self.y_raw = self.y.copy()
        self.y = torch.Tensor(self.y)
        # self.y = self.to_one_hot()
        # self.X = self.get_position_vectors()
        self.X = self.adjust_input_data(self.X)
        # self.split_train_and_test()
        print("ALL FINISHED")

    def n_skel(self):
        # Shape (N, 2, 351, 17, 2)
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                for k in range(len(self.X[i][j])):
                    self.X[i][j][k] = normalize_skeleton(self.X[i][j][k])
        return self.X

    def show_video(self, i, include_centroids=False, max_frames=200, outfile = "plots.gif"):
        # print(self.classes, self.video_filenames)
        # quit()
        for ic, class_name in enumerate(self.classes):
            found_class = False    
            for iv, v in enumerate(self.video_filenames):
                curr_vid_class = v.split("/")[6]
                if not found_class:
                    if curr_vid_class == class_name:
                        found_class = True
                        self._show_video(iv + i, include_centroids, max_frames, outfile=f"{class_name}_plot.gif")




        # pass
        
    def _show_video(self, i, include_centroids=False, max_frames=200, outfile = "plots.gif"):
        i = abs(i)
        print()
        print("Generating Video...")
        step_count = 1
        
        fig, ax = plt.subplots(3)
        cap = self.video_filenames[i]
        frames = []
        ax, vid_ax, vid_scat_ax = ax[0], ax[1], ax[2]
        ax.invert_yaxis()
        camera = Camera(fig)

        frames = cv2.VideoCapture(cap)
        a = 0 
        while True:

            ret, frame = frames.read()
            if not ret:
                break
            curr_frame = []
            kps = self.X[i][0][a]
                      

            coords = [k[:2] for k in kps]
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_ax.imshow(im)
            def plot_points(curr_ax, mul_x = 1, mul_y=1, max_kp=-1):
                curr_ax.scatter(coords[0][0] * mul_x, coords[0][1] * mul_y, c='r')
                curr_ax.scatter([c[0]   * mul_x for c in coords[1:max_kp]], [c[1]  * mul_y for c in coords[1:max_kp]], c='b')
                for c in self.connections:
                    curr_points = [coords[self.kp_indices[con]] for con in c]
                
                    curr_ax.plot(
                        [curr_points[1][0]* mul_x, curr_points[0][0]* mul_x ] , 
                        [curr_points[1][1] * mul_y, curr_points[0][1] * mul_y ], c="g")
            plot_points(ax)
            plot_points(vid_scat_ax, mul_x=im.shape[1], mul_y=im.shape[0])
            vid_scat_ax.imshow(im)
            camera.snap()
            a += 1
        print("Rendering Video")
        animation = camera.animate() 
        print("Writing Video")
        if not os.path.exists("gifs"):
            os.makedirs("gifs")
        path_out = os.path.join("gifs", outfile)
        animation.save(path_out, writer='imagemagick', progress_callback=cell_callback_factory(len(self.X[i][0][:(-1 if len(self.X[i]) < max_frames else max_frames)])))

    def convert_to_one_hot(self):
        onehot_mat = np.zeros((self.y.shape[0], self.y.shape[1], self.n_classes))
        for i in range(onehot_mat.shape[0]):
            for j in range(onehot_mat.shape[1]):
                onehot_mat[i, j, self.y[i, j]] = 1
        return onehot_mat


    def interpolate_by_time(self, convert_to_numpy=True):
        min_frames = self.num_timesteps
        new_x = []
        new_y = []
        for i in range(len(self.X)):
            curr_class = self.y[i]
            curr_vid_len = len(self.X[i][0])
            if self.classes[curr_class] == 'Normal':
                curr_y = [curr_class] * curr_vid_len
            else:
                curr_y = [curr_class] * curr_vid_len
            
            for k in range(curr_vid_len - min_frames):
                new_x.append(self.X[i][0][k:k + min_frames])
                new_y.append(curr_y[k:k + min_frames])
        if convert_to_numpy:
            self.X = np.array(new_x, dtype=np.double)
            self.y = np.array(new_y, dtype=np.int32)

    def split_train_and_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_split)


    def reshape_skeletons(self):
        # Need skeleton to be shape (vid_seq, frame, keypoints, 2)
        reshaped_skel = []
        reshaped_z_data = []
        for i, person in enumerate(self.skel_data):
            # reshaped_skel.append([])
            reshaped_skel.append([[],[]])
            for l in person:
                # quit()
                if l[1] == self.headpoint:
                    reshaped_skel[-1][l[0]].append([])
                reshaped_skel[-1][l[0]][-1].append(l[2:])

        
            
        return reshaped_skel

    def create_file_data(self, kp_dict, max_samples, max_classes):
        if not os.path.exists(os.path.join(self.directory, f"cached/")):
            os.makedirs(os.path.join(self.directory, "cached/"))
        videos = []
        z_data = []
        vid_filenames = []
        classe_names = os.listdir(self.directory)
        classe_names.remove("cached")
        if self.exclude_classes is not None:
            tmp = {}
            for i, c in enumerate(classe_names):
                tmp[c] = i
            for ex in self.exclude_classes:
                # print(f"REMOVIGN {ex}")
                if type(ex) == str:
                    tmp.pop(ex)
                elif type(ex) == int:
                    tmp.popitem(ex)
            
            classe_names = list(tmp.keys())
        if max_classes is not None:
            classe_names = classe_names[:max_classes] if len(classe_names) > max_classes else classe_names
        classes = []

        
        for class_idx ,class_name in enumerate(classe_names):
            print("#################################")
            print(f"Class {class_idx + 1} / {len(classe_names)}")
            print("#################################")
            try:
                vids = os.listdir(
                    os.path.join(self.directory, class_name)
                )
            except Exception as e:
                continue
            
            if max_samples is None:
                maximum_num_samples = int(len(vids))
            elif 0 < max_samples < 1:
                maximum_num_samples = int(len(vids) * max_samples)
            elif max_samples < len(vids):
                maximum_num_samples = max_samples
            else:
                maximum_num_samples = int(len(vids))
            class_vids = []
            z_class_vids = []
            class_vid_filenames = []

            loop = tqdm(vids)
            if not os.path.exists(os.path.join(self.directory, "cached/")):
                os.makedirs(os.path.join(self.directory, "cached/"))
            num_samples = 0
            tmp_z = []
            tmp_vids = []
            tmp_vid_filenames = []
            tmp_classes = []

            for i, vid in enumerate(loop):
                if os.path.exists(os.path.join(self.directory, "cached/", f"{vid}.txt")) and READ_FROM_CACHE:
                    ret_val = read_from_cached_file(os.path.join(self.directory, "cached/", f"{vid}.txt"))
                    
                    if ret_val is not None:
                        num_samples += 1
                        to_append= ret_val
                        to_append = [[ind, a, b, c] for (ind, a, b, c) in to_append]
                        tmp_vids.append(to_append)
                        tmp_vid_filenames.append(os.path.join(self.directory,class_name, vid))
                        tmp_classes.append(class_idx)
                else:
                    print(f"[{class_idx + 1} / {len(classe_names)}]File cached/{vid}.txt doesn't exist, creating")
                    filename = os.path.join(self.directory, class_name, vid)
                    c = get_kp_from_video(filename, kp_dict)
                    if c is not None:
                        num_samples += 1
                        tmp_vids.append(c)
                        tmp_vid_filenames.append(filename)
                        if WRITE_TO_CACHE:
                            write_to_file(os.path.join(self.directory, "cached/", f"{vid}.txt"), c)
                        tmp_classes.append(class_idx)
                    else:
                        if WRITE_TO_CACHE:
                            write_to_file(os.path.join(self.directory, "cached/", f"{vid}.txt"), "")
                
                if num_samples == maximum_num_samples:
                    print("Reached Maximum Samples")
                    break
            if num_samples >= self.min_samples:
                videos.extend(tmp_vids)
                classes.extend(tmp_classes)
                vid_filenames.extend(tmp_vid_filenames)
                z_class_vids.extend(tmp_z)
            
        i = 0
        j = 0
        while(i < len(classe_names)):
            if classes.count(j) == 0:
                print(f"Popping {classe_names[i]} (Class {i})")
                classe_names.pop(i)
                j += 1
            else:
                i += 1
                j += 1

        for i, val in enumerate(np.unique(classes)):
            classes = [i if x==val else x for x in classes]
       

        self.video_filenames = vid_filenames
        self.classes = classe_names
        return videos, kp_dict, classes


    def _get_file_data(self, max_samples, max_classes):
        keypoints_arr = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        kp_dict = {}
        for i, s in enumerate(keypoints_arr):
            kp_dict[s] = i
        return self.create_file_data(kp_dict, max_samples, max_classes)
        

    def setup_information(self):               
        self.connections = [
            ('nose', 'left_eye'),
            ('nose', 'right_eye'),
            ('left_eye', 'left_ear'),
            ('right_eye', 'right_ear'),
            ('left_ear', 'left_shoulder'),
            ('right_ear', 'right_shoulder'),
            ('right_shoulder', 'left_shoulder'),
            

            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),

            
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),

            ('right_shoulder', 'right_hip'),
            ('left_shoulder', 'left_hip'),
            ('right_hip', 'left_hip'),

            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),

            
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),

        ]

        self.edge_matrix = [[self.kp_indices[x] for x, _ in self.connections], [self.kp_indices[y] for _, y in self.connections]]
        self.in_edge = [
            (self.kp_indices[x], self.kp_indices[y])
            for (x, y) in self.connections
        ]        

        self.upper_torso = [
            "nose",
            "right_eye",
            "right_ear",
            
            "left_eye",
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
        

        

    def adjust_input_data(self, X):
        N = 0
        C = 4
        T = 2
        V = 3
        M = 1
        X = X.reshape((X.shape[0],) + (1,) + X.shape[1:]).transpose(N, C, T, V, M)
        if X.shape[-1] > 1:
            X = X[..., :1]
        return torch.tensor(X, dtype=torch.float32)