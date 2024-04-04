import os
import cv2
import glob

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .genericdataset import GenericGaitDataset
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import torch
from scipy.interpolate import interp1d
# matplotlib.use('TkAgg')
np.seterr(all='raise')

from sklearn.ensemble import RandomForestClassifier
from celluloid import Camera

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8, smooth_landmarks=True)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

READ_FROM_CACHE = True
WRITE_TO_CACHE = True

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    with open(filename, 'r') as in_file:
        lines = in_file.read().split("\n")
    curr_data = []
    z_data = []
    for line in lines:
        if line == "": continue
        i, kp, x, y, z = line.split(";")
        curr_data.append([int(i), kp, float(x), float(y)])
        z_data.append([float(z)])
    if curr_data == [] or (len(curr_data) // 33) < min_samples:
        return None
    return curr_data, z_data
    
        

def get_kp_from_file(filename, kp_dict, min_frames = 2, im_height=256, im_width=256):
    # TODO
    
    frames = os.listdir(filename)

    frame_results = []
    for frame in frames:
        curr_frame = []
        # image = mp.Image.create_from_file(
        #     os.path.join(filename, frame)
        # ).numpy_view()
        image = cv2.imread(
            os.path.join(filename, frame)
        )
        image = cv2.resize(image, (im_height, im_width), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
        if len(boxes) == 0:
            continue
        num_boxes = 2 if len(boxes) > 2 else 1
        boxes = boxes[:num_boxes]
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        found_frame = False
        for i in range(2):
            if i >= len(boxes):
                for k in kp_dict.keys():
                    curr_frame.append([i, k, 0, 0, 0])
                continue
            xA, yA, xB, yB = boxes[i]
            cropped = image[yA:yB, xA:xB]

            pose_results = pose.process(cropped)
            # print(dir(pose_results))
            # quit()
            if pose_results.pose_landmarks is not None:
                found_frame = True
                for k in kp_dict.keys():
                    v = kp_dict[k]
                    curr_frame.append([i, k, pose_results.pose_landmarks.landmark[v].x, pose_results.pose_landmarks.landmark[v].y, pose_results.pose_landmarks.landmark[v].z])
                    # print(dir(pose_results.pose_world_landmarks))
                    # quit()
        if found_frame:
            frame_results.extend(curr_frame)
    if len(frame_results) < (min_frames*len(kp_dict.keys()) * 2):
        print(f"{filename} has less than {min_frames} frames, skipping")
        return None
    return frame_results




def write_to_file(filename, kps):
    lines = []
    if kps == "":
        with open(filename, "w") as outfile:
            outfile.write("")
        return
    for kp in kps:
        curr_line = [";".join([str(x) for x in kp])]
        # for p in kp:
        #     c = []
        #     for i in p:
        #         c.append(str(i))
        #     curr_line.append(";".join(c))
        lines.extend(curr_line)
    with open(filename, "w") as outfile:
        outfile.write("\n".join(lines))

class HMDBDataset(GenericGaitDataset):

    def __init__(self, directory='../../../Datasets/HMDB51/HMDB51/', max_samples=None, t_interp=6, num_dims=2, generate_test_video=None, extract_steps=False, test_split=0.1, val_split=0.3, max_classes=None, num_timesteps=12, exclude_classes=None):
        
        super().__init__(directory=directory, max_samples=max_samples, t_interp=t_interp, num_dims=num_dims, generate_test_video=generate_test_video, extract_steps=extract_steps)
        self.val_split = val_split
        self.exclude_classes = exclude_classes
        self.test_split = test_split
        self.max_classes = max_classes
        self.num_timesteps = num_timesteps
        self.initialise_stuff()

    def initialise_stuff(self):
        self.skel_data, self.kp_indices, self.labels = self._get_file_data(self.max_samples, self.max_classes) # (n_people, n_files, n_lines, 3)
        # print(self.skel_data)
        # quit()
        
        
        
        
        self.setup_information()

        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        
        
        self.X = self.reshape_skeletons()
        # print(dim(self.X))
        # quit()
        
        
        self.y  = self.labels

        # print(self.X[1][0])
        # quit()
        if self.generate_test_video is not None:
            self.show_video(self.generate_test_video)
        if self.num_dims > 2:
            self.rotation_vector(2)
            if self.generate_test_video is not None:
                self.show_video(self.generate_test_video)
        if self.generate_test_video is not None and self.generate_test_video < 0:
            quit()
   

        # self.translation_vector()
        # self.scaling_vector()
        # quit()     
        # self.X, self.y = self.get_individual_steps()
   
        self.interpolate_by_time()
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
        # self.y = self.to_one_hot()
        # self.X = self.get_position_vectors()
        self.X = self.adjust_input_data(self.X)
        self.split_train_and_test()

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
        loop = tqdm(range(len(self.X[i][:(-1 if len(self.X[i]) < max_frames else max_frames)])))
        # print(dim(loop))
        # quit()

        # ankles, l, r = self.ankle_distances(i, return_positions=True)
        # peaks = self._find_peaks_for_video(i)[::2]
        step_count = 1
        
        fig, ax = plt.subplots(3)
        # ax = fig.add_subplot()
        # ankle_ax = fig.add_subplot()
        cap = [os.path.join(self.video_filenames[i], f) for f in os.listdir(self.video_filenames[i])]
        # print(cap)
        # quit()
        if len(cap) == 0:
            print(f"Error opening file {self.video_filenames[i]}")
            quit()
        frames = []
        ax, vid_ax, vid_scat_ax = ax[0], ax[1], ax[2]
        ax.invert_yaxis()
        camera = Camera(fig)
        
        for a in loop:
            print(a)
            # if peaks[(step_count - 1) % len(peaks)] < a < peaks[(step_count) % len(peaks)] :
            #     step_count += 1
            curr_frame = []
            # print(dim(self.X[i]))
            # quit()
            kps = self.X[i][a]
            # ax.legend([f"Step Count: {step_count}"], loc='upper left')
            # ankle_ax.plot(ankles[:a])
            

            coords = [k[:2] for k in kps]#[kps[a::self.num_dims] for a in range(self.num_dims)]
            


            # lines = [[]] * len(coords)
            # for c in self.connections:
            #     # print(self.kp_indices)
            #     # print(a)
            #     # quit()
            #     # print(dim(coords))
            #     # quit()

            #     curr_points = [coords[self.kp_indices[con]] for con in c]
            #     # print(dim(curr_points))
            #     # quit()

            #     ax.plot(curr_points[0], curr_points[1])

                
                # curr_frame.append(ax.plot(xline, yline, c='b'))
            # print(*lines)
            # quit()
            # ax.plot(lines[0], lines[1])
            # for k in range(len(lines[0])):

            #     ax.plot([lines[0][k]], , c='b')

            
            # curr_frame.append(ax.scatter(x, y))
            # frames.append([ax.scatter(x,y)])
            # print(*lines[:2])
            # quit()
            # quit()
            # ret, frame = cap.read()
            print(cap[a])
            # quit()
            frame = cv2.imread(cap[a])
            # print(frame)
            # quit()
            # cv2.imshow(frame)
            # cv2.waitkey(0)
            # if not ret:
            #     print("There was an error")
            #     quit()
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vid_ax.imshow(im)
            # print(dir(im))
            # print(im.shape)
            # quit()
            # print([c[:2] for c in coords])
            # print(self.X[i][a])
            # quit()
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
            # ax.invert_yaxis()
            # ax.scatter(coords[0][0], coords[0][1], c='r')
            # ax.scatter([c[0] for c in coords[1:]], [c[1] for c in coords[1:]], c='b')
            # vid_scat_ax.scatter(coords[0][0], coords[0][1], c='r')
            # vid_scat_ax.scatter([c[0] * im.shape[0] for c in coords[1:]], [c[1] * im.shape[1] for c in coords[1:]], c='b')
            vid_scat_ax.imshow(im)
            camera.snap()
            loop.set_postfix()
        # fig.savefig("3d.png")
        print("Rendering Video")
        # ani = animation.ArtistAnimation(fig, frames, interval=50)
        animation = camera.animate() 
        print("Writing Video")
        if not os.path.exists("gifs"):
            os.makedirs("gifs")
        path_out = os.path.join("gifs", outfile)
        animation.save(path_out, writer='imagemagick', progress_callback=cell_callback_factory(len(self.X[i][:(-1 if len(self.X[i]) < max_frames else max_frames)])))
        # ani.save('movie.mp4')
        # quit()

    def convert_to_one_hot(self):
        onehot_mat = np.zeros((len(self.y), self.n_classes))
        # print(onehot_mat.shape)
        # print(max(self.y))
        # print(np.unique(self.y))
        # quit()
        for i in range(onehot_mat.shape[0]):
            # print(i, dim(self.y), self.y)
            onehot_mat[i, self.y[i]] = 1
        return onehot_mat


    def interpolate_by_time(self, convert_to_numpy=True):
        min_frames = self.num_timesteps#max([len(x) for x in self.X])//4
        # print(dim(self.X))
        # print([len(x) for x in self.X])
        # quit()
        for i in range(len(self.X)):
            for j in range(len(self.X[i])):
                curr_sample = self.X[i][j]
                x = np.arange(len(curr_sample))
                # print(len(curr_sample), i, len(self.X))
                print(dim(curr_sample), i, j)
                # quit()
                f = interp1d(x, curr_sample, axis=0)
                xnew = np.linspace(0, len(curr_sample) - 1, min_frames)
                # print(xnew, len(curr_sample))
                # quit()
                # y_old = f(x)
                print(xnew, len(curr_sample))
                # quit()
                ynew = f(xnew)
                self.X[i][j] = ynew
        if convert_to_numpy:
            self.X = np.array(self.X, dtype=np.double)

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
        # print(upper_limbs)
        # quit()
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
        # classes = []
        videos = []
        z_data = []
        vid_filenames = []
        classe_names = os.listdir(self.directory)
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
        # print(len(classe_names))
        if max_classes is not None:
            classe_names = classe_names[:max_classes] if len(classe_names) > max_classes else classe_names
        # print(classe_names, len(classe_names), max_classes)
        # quit()
        classes = []

        
        for class_idx ,class_name in enumerate(classe_names):
            print("#################################")
            print(f"Class {class_idx + 1} / {len(classe_names)}")
            print("#################################")
            vids = os.listdir(
                os.path.join(self.directory, class_name)
            )
            
            if max_samples is None:
                pass
            elif 0 < max_samples < 1:
                vids = vids[:int(len(vids) * max_samples)]
            elif max_samples < len(vids):
                vids = vids[:int(max_samples)]
            # classes.append(person_id)
            class_vids = []
            z_class_vids = []
            class_vid_filenames = []

            loop = tqdm(vids)
            if not os.path.exists(f"cached/"):
                os.makedirs(f"cached/")

            for i, vid in enumerate(loop):
                if os.path.exists(f"cached/{vid}.txt") and READ_FROM_CACHE:
                    # to_append, z_datum = read_from_cached_file(f"cached/{vid}.txt")
                    ret_val = read_from_cached_file(f"cached/{vid}.txt")
                    
                    # if ret_val is None:
                    #     continue
                    # to_append, z_datum = ret_val
                    # to_append = [[a, b, c, d[0]] for ((a, b, c), (d)) in zip(to_append, z_datum)]
                    # to_append = np.concatenate((to_append, z_datum), axis=1)
                  
                    # quit()
                    if ret_val is not None:
                        to_append, z_datum = ret_val
                        to_append = [[ind, a, b, c, d[0]] for ((ind, a, b, c), (d)) in zip(to_append, z_datum)]
                        videos.append(to_append)
                        z_class_vids.append(z_datum)
                        vid_filenames.append(os.path.join(self.directory,class_name, vid))
                        classes.append(class_idx)
                else:
                    print(f"[{class_idx + 1} / {len(classe_names)}]File cached/{vid}.txt doesn't exist, creating")
                    filename = os.path.join(self.directory, class_name, vid)
                    c = get_kp_from_file(filename, kp_dict)
                    # print(c)
                    # quit()
                    if c is not None:
                        videos.append(c)
                        vid_filenames.append(filename)
                        if WRITE_TO_CACHE:
                            write_to_file(f"cached/{vid}.txt", c)
                        classes.append(class_idx)
                    else:
                        # pass
                        if WRITE_TO_CACHE:
                            write_to_file(f"cached/{vid}.txt", "")
            # videos.append(class_vids)
            # z_data.append(z_class_vids)
            # print([len(v)//33 for v in videos])

        # print(dim(videos), dim(classes))
        # print([len(v)//33 for v in videos])
        # quit()
        # print(classes, classe_names)
        # quit()
        i = 0
        j = 0
        while(i < len(classe_names)):
            if classes.count(j) == 0:
                print(f"Popping {i}")
                classe_names.pop(i)
                j += 1
            else:
                i += 1
                j += 1

        for i, val in enumerate(np.unique(classes)):
            classes = [i if x==val else x for x in classes]
       

        self.video_filenames = vid_filenames
        # print(self.video_filenames)
        # quit()
        self.classes = classe_names
        return videos, kp_dict, classes
            

    def _get_file_data(self, max_samples, max_classes):
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
        return self.create_file_data(kp_dict, max_samples, max_classes)
        

    def setup_information(self):               
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
        self.in_edge = [
            (self.kp_indices[x], self.kp_indices[y])
            for (x, y) in self.connections
        ]        

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
        

        

    def adjust_input_data(self, X):
        X = X[...,:-1] # Remove Z Dimension
        # X = X.reshape(X.shape + (1,))
        # print(X.shape)
        N = 0
        C = 4
        T = 2
        V = 3
        M = 1
        # print(X.shape)
        # quit()
        X = X.transpose(N, C, T, V, M)
        # print(X.shape)
        return torch.tensor(X, dtype=torch.double)