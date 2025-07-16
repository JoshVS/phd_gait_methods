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
    Reads keypoints from a cached file written in the ShopLiftingDataset format.

    Args:
        filename (str): Path to the cached keypoints file.

    Returns:
        list: Nested list of shape (timesteps, M, 17, 4), where each entry contains
                [person_index, keypoint_name, x, y].
    """
    kps = []
    guilty_people = []
    with open(filename, "r") as infile:
        lines = infile.read().splitlines()
    if len(lines) == 0:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            guilty_people.append(float(line))
            continue
        except Exception as e:
            pass
        
        timestep = []
        people = line.split("/")
        for person in people:
            if not person.strip():
                continue
            keypoints = []
            kp_entries = person.split(";")
            for kp in kp_entries:
                if not kp.strip():
                    continue
                parts = kp.split(",")
                if len(parts) != 4:
                    continue
                i, k, x, y = parts
                keypoints.append([int(i), k, float(x), float(y)])
            timestep.append(keypoints)
        kps.append(timestep)
    return kps, guilty_people
    

def kp_from_frame(frame, kp_dict, annotations=None, show=False, return_bbox=False):
    """
    Extracts keypoints from a single frame using a YOLOv8 pose model.

    Returns list of shape (M * 17, 4) where M is the number of people detected in the frame.
    """
    found_frame = False
    curr_frame = []
    
    try:
        results_generator = yolo_model(source=frame, show=show, conf=0.15, save=False, stream=True, verbose=False)
    except Exception as e:
        return None

    bboxes = []
    if annotations is not None:
        did_shoplifting = []
        time, x, y = annotations
    for i, res in enumerate(results_generator):
        
        kpts = res.keypoints.xy.cpu().numpy() # Shape (17, 2)
        if kpts.shape[1] == 0:
            continue
        check_boxes = res.boxes.xyxyn.cpu().numpy()
        for j in range(kpts.shape[0]):
            curr_frame.append([])
            found_frame = True  

            
            bbox = check_boxes[j]
            xc1, yc1, xc2, yc2 = check_boxes[j,...]
            if annotations is not None:
                if point_is_in_box((x, y), (xc1, yc1, xc2, yc2)):
                    did_shoplifting.append(time)
                else:
                    did_shoplifting.append(-1)
            bboxes.append(bbox)
            xA, yA, w, h = bbox
            for k in kp_dict.keys():
                v = kp_dict[k]
                curr_frame[-1].append([i, 
                                   k, 
                                   (kpts[j, v, 0] - xA) / w, 
                                   (yA - kpts[j, v, 1]) / h]
                )
    if found_frame:
        if annotations is not None:
            curr_frame = (curr_frame, did_shoplifting)
        if return_bbox:
            return curr_frame, bboxes
        return curr_frame
    else:
        return None

def point_is_in_box(point, box):
    x1, x2, y1, y2 = box
    xp, yp = point
    return (x1 <= xp <= x2) and (y1 <= yp <= y2)

def get_kp_from_video(filename, kp_dict, annotations=None, min_frames = 2, im_height=256, im_width=256):    
    """
    Reads in a video file and extracts keypoints using a YOLOv8 pose model.

    Args:
        filename : The filename of the video to be processed. 
        kp_dict: A dictionary mapping keypoint names to their indices. There should be 17 keypoints, and kp_dict[k] = index of keypoint k.

    Returns:
        list :  list is of shape (timesteps, M, 17, 4). Each timestep contains 17 keypoints, and each 
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

    frame_results = []
    if annotations is not None:
        people_shoplfting = []

    for  frame in frames:
        curr_frame = kp_from_frame(frame, kp_dict, annotations=annotations)
        if curr_frame is not None:
            if annotations is not None:
                curr_frame, did_shoplifting = curr_frame
                if len(did_shoplifting) > len(people_shoplfting):
                    people_shoplfting += [[]] * (len(did_shoplifting) - len(people_shoplfting))
                for i in range(len(did_shoplifting)):
                    people_shoplfting[i].append(did_shoplifting[i])
            if len(curr_frame) > len(frame_results):
                frame_results.extend([[]] * (len(curr_frame) - len(frame_results)))
            for i in range(len(curr_frame)):
                frame_results[i].append(curr_frame[i])
    if len(frame_results) < 1:
        print(f"{filename} has no people, skipping")
        return None
    if annotations is not None:
        frame_results = (frame_results, people_shoplfting)
    return frame_results


def write_to_file(filename, kps, time=None, guilty=None):
    """
    Writes keypoints to a cached file

    Args:
        filename : The filename to be written to.
        kps : A list of shape (M, timesteps, 17, 4). Each timestep contains 17 keypoints, and each keypoint contains i (person index), k (keypoint name), x and y (normalized to the bounding box).

    Returns:
        Nothing
    """
    
    lines = []
    if kps == "":
        with open(filename, "w") as outfile:
            outfile.write("")
        return
    for i, t in enumerate(kps):
        if i is not None:
            if i == guilty:
                lines.append(str(time))
            else:
                lines.append(str(-1))
        else:
            lines.append(str(-1))
        ilines = []
        for m in t:
            jlines = []
            for k in m:
                jlines.append(",".join([str(p) for p in k]))
            ilines.append(";".join(jlines))
        lines.append("/".join(ilines))
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines))


class ShopLiftingDataset(GenericGaitDataset):
    """
    A dataset loader for the ShopliftingDataset, designed for gait analysis and classification tasks.
    This class extends `GenericGaitDataset` and provides methods for loading, preprocessing, normalizing,
    and splitting skeleton-based gait data extracted from video files. It supports caching, class exclusion,
    time interpolation, and visualization of skeletons over video frames.
    Args:
        directory (str): Path to the dataset root directory. Defaults to '../../../Datasets/ShopliftingDataset/Dataset/'.
        max_samples (int or float, optional): Maximum number of samples to load per class. If float in (0,1), interpreted as a fraction.
        min_samples (int): Minimum number of samples required per class to include it. Defaults to 5.
        t_interp (int): Interpolation factor for time dimension. Defaults to 6.
        num_dims (int): Number of spatial dimensions for keypoints (usually 2 or 3). Defaults to 2.
        generate_test_video (int, optional): If set, generates a visualization for the specified video index.
        extract_steps (bool): Whether to extract step information from skeletons. Defaults to False.
        test_split (float): Fraction of data to use as test set. Defaults to 0.1.
        val_split (float): Fraction of training data to use as validation set. Defaults to 0.3.
        max_classes (int, optional): Maximum number of classes to load.
        num_timesteps (int): Number of frames per sample after time interpolation. Defaults to 12.
        exclude_classes (list, optional): List of class names or indices to exclude from the dataset.
    Attributes:
        skel_data (list): Loaded skeleton data for all samples.
        kp_indices (dict): Mapping from keypoint names to indices.
        labels (list): List of class labels for each sample.
        val_split (float): Validation split ratio.
        min_samples (int): Minimum samples per class.
        exclude_classes (list): Classes to exclude.
        test_split (float): Test split ratio.
        max_classes (int): Maximum number of classes.
        num_timesteps (int): Number of frames per sample.
        video_filenames (list): List of video file paths corresponding to samples.
        classes (list): List of class names.
        X (np.ndarray or torch.Tensor): Preprocessed input data.
        y (np.ndarray or torch.Tensor): One-hot encoded class labels.
        n_classes (int): Number of unique classes.
        edge_matrix (list): Edge connections for skeleton graph.
        connections (list): List of keypoint connections for visualization.
        headpoint (str): Name of the head keypoint.
        ... (other keypoint names as attributes)
    Methods:
        initialise_stuff(): Loads and preprocesses the dataset, normalizes skeletons, interpolates by time, and prepares labels.
        n_skel(): Normalizes skeletons for each sample.
        show_video(i, include_centroids=False, max_frames=200, outfile="plots.gif"): Visualizes skeletons overlaid on video frames.
        _show_video(i, include_centroids=False, max_frames=200, outfile="plots.gif"): Helper for show_video.
        convert_to_one_hot(): Converts class labels to one-hot encoding.
        interpolate_by_time(convert_to_numpy=True): Interpolates skeleton sequences to fixed length.
        split_train_and_test(): Splits data into training, validation, and test sets.
        reshape_skeletons(): Reshapes raw skeleton data into (video, frame, keypoints, 2) format.
        create_file_data(kp_dict, max_samples, max_classes): Loads skeleton data from files, applies caching, and filters classes.
        _get_file_data(max_samples, max_classes): Prepares keypoint dictionary and loads data.
        setup_information(): Sets up skeleton graph connections and keypoint indices.
        create_edge_matrix(): Creates edge matrix for skeleton graph.
        adjust_input_data(X): Adjusts input data shape for model compatibility.
    Example:
        dataset = ShopLiftingDataset(directory='path/to/data', max_samples=100, num_timesteps=12)
    """
    

    def __init__(self, directory='../../../Datasets/ShopliftingDataset/Dataset/', max_samples=None, min_samples=5, t_interp=6, num_dims=2, generate_test_video=None, extract_steps=False, test_split=0.1, val_split=0.3, max_classes=None, num_timesteps=12, exclude_classes=None, skip_frames=1):
        
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video, extract_steps=extract_steps)
        self.skip_frames = skip_frames
        self.val_split = val_split
        self.min_samples = min_samples
        self.exclude_classes = exclude_classes
        self.test_split = test_split
        self.max_classes = max_classes
        self.num_timesteps = num_timesteps
        self.initialise_stuff()

    def initialise_stuff(self):
        torch.set_default_dtype(torch.float32)
        self.curr_people = []
        
        self.kp_indices = self._get_file_data(self.max_samples, self.max_classes) 
        if self.directory is not None:
            self.open_annotations(os.path.join(self.directory, "annotations.txt"))
            self.skel_data, self.labels = self.create_file_data(self.kp_indices, self.max_samples, self.max_classes)
        
        
        
        self.setup_information()

        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        
        if self.directory is not None:
            print("RESHAPING AND NORMALIZING SKELETONS")
            self.X = self.reshape_skeletons()
            self.X = self.n_skel()
            
        
            self.y  = self.labels

        if self.generate_test_video is not None:
            self.show_video(self.generate_test_video)
   

        if self.directory is not None:
            print("INTERPOLATING BY TIME")

            self.interpolate_by_time()
            self.adjust_classes(0.1)
            self.stratify_y = self.y.copy()
            y_unique, counts = np.unique(self.y, return_counts=True)
            plt.figure()
            plt.bar(self.classes, counts)
            plt.xticks(rotation=90)
            plt.savefig("class_dist.png")
            plt.close()


        
            self.n_classes = len(np.unique(self.y))
            self.y = self.convert_to_one_hot()
        
            self.y_raw = self.y.copy()
            self.y = torch.Tensor(self.y)
            self.X = self.adjust_input_data(self.X)
            print("DONE")

    def adjust_classes(self, ratio):        
        y_sum = np.sum(self.y, axis=1)
        y_sum[y_sum > 0] = 1
        y_unique, counts = np.unique(y_sum, return_counts=True)
        max_class = y_unique[np.argmax(counts)]
        y_first, = np.where(y_sum==max_class)   
        num_y_samples = int(min(counts) * ratio)
        to_delete = np.random.choice(y_first, size=(max(counts) - num_y_samples))
        self.y = np.delete(self.y, to_delete, axis=0)
        self.X = np.delete(self.X, to_delete, axis=0)



    def open_annotations(self, filename):
        with open(filename) as f:
            lines = f.readlines()

        self.shoplifting_data = {}

        for line in lines:
            fname, start_frame, x, y = line.split(",")
            self.shoplifting_data[fname] = (float(start_frame), float(x), float(y))

    def add_frame(self, frame):
        kp_results = kp_from_frame(frame, self.kp_indices, show=True, return_bbox=True)
        if kp_results is None:
            return
        
        curr_frame, curr_bboxes = kp_results
        
        self.bboxes = curr_bboxes
        if len(curr_frame)  > len(self.curr_people):
            # Update X to match number of people
            self.curr_people.extend([[]] * (len(curr_frame)  - len(self.curr_people)))
        for i in range(len(curr_frame)):
            # self.curr_people[i].append(curr_frame[i])
            # curr_person = []
            # for j in range(len(curr_frame)):
            #     if curr_frame[j][0] == i:
            #         curr_person.append(curr_frame[j])               


            if len(self.curr_people[i]) < self.num_timesteps:
                self.curr_people[i].append(curr_frame[i])
            else:
                self.curr_people[i] = self.curr_people[i][1:] + [curr_frame[i]]

        self.X = []
        for i in range(len(self.curr_people)):
            if len(self.curr_people[i]) == self.num_timesteps:
                self.X.append(self.curr_people[i])

        if len(self.X) > 0:
            self.X = self.reshape_skeletons(skel_data=self.X, unmash_kp=False)
            self.X = self.n_skel(streaming=True)
            self.X = np.array(self.X, dtype=np.float32)
            self.X = self.adjust_input_data(self.X)

    def get_skeletons(self, skel_ids=None):
        if skel_ids is None:
            return self.X
        else:
            return self.X[skel_ids]
    
    def has_people(self):
        return len(self.X) > 0

        

    def n_skel(self, streaming=True):
        # Shape (N, 2, 351, 17, 2)
        if not streaming:
            for i in range(len(self.X)):
                for j in range(len(self.X[i])):
                    for k in range(len(self.X[i][j])):
                        self.X[i][j][k] = normalize_skeleton(self.X[i][j][k])
            return self.X
        else:
            for i in range(len(self.X)):
                for j in range(len(self.X[i])):
                    self.X[i][j] = normalize_skeleton(self.X[i][j])
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
        min_frames = self.num_timesteps * self.skip_frames
        new_x = []
        new_y = []
        loop = tqdm(range(len(self.X)))
        for i in loop:
            curr_class = self.y[i]
            curr_vid_len = len(self.X[i])
            if self.guilty_people[i] == -1:           
                curr_y = [0] * curr_vid_len
            else:
                curr_y = [0] * int(curr_vid_len * self.guilty_people[i])
                curr_y += [1] * (curr_vid_len - len(curr_y))
            loop.set_postfix()
            
            for k in range(curr_vid_len - min_frames):
                new_x.append(self.X[i][k:k + min_frames:self.skip_frames])
                new_y.append(curr_y[k:k + min_frames:self.skip_frames])
        if convert_to_numpy:
            self.X = np.array(new_x, dtype=np.double)
            self.y = np.array(new_y, dtype=np.int32)
            
    def split_train_and_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=self.val_split)


    def reshape_skeletons(self, skel_data=None, unmash_kp = False):
        if skel_data is None:
            skel_data = self.skel_data
        # Need skeleton to be shape (vid_seq, frame, keypoints, 2)
        reshaped_skel = []
        reshaped_z_data = []
        for i, person in enumerate(skel_data):
            # reshaped_skel.append([])
            if unmash_kp:
                reshaped_skel.append([[],[]])
                for l in person:
                    # quit()
                    if l[1] == self.headpoint:
                        reshaped_skel[-1][l[0]].append([])
                    reshaped_skel[-1][l[0]][-1].append(l[2:])

            else:
                reshaped_skel.append([])
                for t in person:
                    reshaped_skel[-1].append([])
                    for k in t:
                        reshaped_skel[-1][-1].append(k[2:])

        return reshaped_skel

    def create_file_data(self, kp_dict, max_samples, max_classes):
        if not os.path.exists(os.path.join(self.directory, f"cached/")):
            os.makedirs(os.path.join(self.directory, "cached/"))
        videos = []
        z_data = []
        vid_filenames = []
        global_guilty_people = []
        classe_names = os.listdir(self.directory)
        classe_names.remove("cached")
        classe_names.remove("annotations.txt")
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
            guilty_people = []
            tmp_classes = []

            for i, vid in enumerate(loop):
                if os.path.exists(os.path.join(self.directory, "cached/", f"{vid}.txt")) and READ_FROM_CACHE:
                    ret_val = read_from_cached_file(os.path.join(self.directory, "cached/", f"{vid}.txt"))
                    if ret_val is not None:
                        ret_val, g = ret_val
                    
                        num_samples += 1#len(ret_val)
                        to_append= ret_val
                        tmp_vids.extend(to_append)
                        tmp_vid_filenames.extend([os.path.join(self.directory,class_name, vid)] * len(to_append))
                        tmp_classes.extend([0] * len(to_append))
                        guilty_people += g
                        
                else:
                    print(f"[{class_idx + 1} / {len(classe_names)}]File cached/{vid}.txt doesn't exist, creating")
                    filename = os.path.join(self.directory, class_name, vid)
                    if class_name.lower() == "shoplifting":
                        c = get_kp_from_video(filename, kp_dict, annotations=self.shoplifting_data[vid])
                    else:
                        c = get_kp_from_video(filename, kp_dict)
                    if c is not None:
                        if class_name.lower()=="shoplifting":
                            c, annot_data = c
                        num_samples += 1
                        tmp_vids.extend(c)
                        tmp_vid_filenames.append(filename)
                        if class_name.lower() == "shoplifting":
                            sums = []
                            for s in range(len(annot_data)):
                                sums.append(sum(annot_data[s]))
                            guilty_person = np.argmax(sums)
                            for g in range(guilty_person):
                                guilty_people.append(-1)
                            guilty_people.append(self.shoplifting_data[vid][0])
                            for g in range(guilty_person + 1, len(sums)):
                                guilty_people.append(-1)
                            tmp_classes.extend([0] * len(c))
                        else:
                            tmp_classes.extend([0] * len(c))
                            guilty_people.extend([-1] * len(c))
                        if WRITE_TO_CACHE:
                            if class_name.lower() == "shoplifting":
                                write_to_file(os.path.join(self.directory, "cached/", f"{vid}.txt"), c, time=self.shoplifting_data[vid][0], guilty=guilty_person)
                            else:
                                write_to_file(os.path.join(self.directory, "cached/", f"{vid}.txt"), c)
                    else:
                        if WRITE_TO_CACHE:
                            write_to_file(os.path.join(self.directory, "cached/", f"{vid}.txt"), "")
                if num_samples >= maximum_num_samples:
                    print("Reached Maximum Samples")
                    break
            if num_samples >= self.min_samples:
                videos.extend(tmp_vids)
                classes.extend(tmp_classes)
                vid_filenames.extend(tmp_vid_filenames)
                z_class_vids.extend(tmp_z)
                global_guilty_people.extend(guilty_people)
            
        # i = 0
        # j = 0
        # while(i < len(classe_names)):
        #     if classes.count(j) == 0:
        #         print(f"Popping {classe_names[i]} (Class {i})")
        #         classe_names.pop(i)
        #         j += 1
        #     else:
        #         i += 1
        #         j += 1

        # for i, val in enumerate(np.unique(classes)):
        #     classes = [i if x==val else x for x in classes]
       

        self.video_filenames = vid_filenames
        self.classes = classe_names
        self.guilty_people = global_guilty_people
        return videos, classes


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
        return kp_dict
        return 
        

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