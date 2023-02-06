import os

import numpy as np
import cv2


from scipy.signal import savgol_filter, find_peaks

from tqdm import tqdm
import matplotlib.pyplot as plt

import mediapipe as mp
mp_pose = mp.solutions.pose

from preprocessing import preprocess_dataset, dist, global_coordinate_frame
from kinect_preprocessing import preprocess_kinect_dataset, global_coordinate_frame_kinect

def _dim(l):
    if not type(l) == list:
        return []
    return [len(l)] + _dim(l[0])

def dim(l):
    return tuple(_dim(l))

def distance(point1, point2):
    x1, y1 = np.array(point1[:][0]), np.array(point1[:][1])
    x2, y2 = np.array(point2[:][0]), np.array(point2[:][1])
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class VideoDataset:

    def __init__(self, label_dir="labels/", info_dir="info/", max_samples = None):
        # skeletons = (video_id, frame_id, kp_id, 2)
        self.label_dir = label_dir
        self.info_dir = info_dir
        self.C_min = 1000000000
        self.C_max = 0
        self.vid_skeletons, self.labels = self._read_skeletons(max_samples)
        self.n_classes = len(np.unique(self.labels))
        self.norm_skeletons = self.normalize_skeletons()
        s = self.norm_skeletons
        # print(len(s[0][0]))
        # quit()
        # self.distances = self.get_distances()
        self.vid_features = self.extract_features()
        v = self.vid_features
        # C_min, C_max = np.min(np.array(self.vid_skeletons).flatten()), np.max(np.array(self.vid_skeletons).flatten())
        self.norm_const = self.C_max - self.C_min

        self.gait_cycles = self.get_gait_cycles()
        self.gait_phases = self.get_gait_phases()
        self.compress_gait_phases()

    def one_hot_labels(self):
        un_labels = list(np.unique(self.labels))
        self.n_classes = len(un_labels)
        onehot = np.zeros((len(self.labels), len(un_labels)))
        for i, l in enumerate(self.labels):
            l_ind = un_labels.index(l)
            onehot[i, l_ind] = 1
        return onehot


    def compress_gait_phases(self):
        ret_val = []
        label_ret = []
        for sample, label in zip(self.gait_phases, self.labels):
            ret_val.extend(sample)
            label_ret.extend([label] * len(sample))
        self.labels = label_ret
        self.gait_phases = ret_val

    def get_distances(self):
        dist = []
        for i, point in enumerate(self.vid_skeletons[0][1:]):
            dist.append(distance(point, self.vid_skeletons[0][i]))
        return dist


    def get_gait_phases(self):
        gait_phases = [
            (0, 10),
            (10, 30),
            (30, 50),
            (50, 60),
            (60, 73),
            (73, 87),
            (87, 100)
        ]
        final_gait_features = []
        for i, v in enumerate(self.gait_cycles):
            vid_g_f = []
            if len(v) == 0:
                self.labels.pop(i)
                continue
            prev_g = v[0]
            for g in v[1:]:
                len_gait = g - prev_g
                curr_gp = []
                for gp in gait_phases:
                    start, stop = gp
                    start = prev_g + int(start / 100 * len_gait)
                    stop = prev_g + int(stop / 100 * len_gait)
                    curr_f = []
                    for f in range(5):
                        curr_vid = self.vid_features[i]
                        curr_frames = curr_vid[start:stop]
                        tmp = None
                        for fr in curr_frames:
                            tmp = fr[f] if tmp is None else [x + y for x, y in zip(fr[f], tmp)]
                        tmp = np.divide(tmp, stop - start)
                        curr_f.extend(tmp)
                    
                    curr_gp.append(curr_f)
                vid_g_f.append(curr_gp)
                prev_g = g
            final_gait_features.append(vid_g_f)
        return final_gait_features

    def extract_features(self):
        features = []
        for i, vid in enumerate(self.norm_skeletons): 
            prev_skel = None
            frame_skels = []
            loop = tqdm(vid)
            for skel in loop:
                frame_skels.append(preprocess_dataset(skel, prev_skel))
                prev_skel = skel
                loop.set_postfix(vid_number=i+1)
            features.append(frame_skels)                
        return features

    def normalize_skeletons(self):
        norm_skel = []
        for vid in self.vid_skeletons:
            vid_skel = []
            for frame in vid:
                point1 = frame[np.argmax(frame[:][1])]
                point2 = frame[np.argmin(frame[:][1])]
                norm_dist = dist(point1, point2)
                norm_kp = []
                for kp in frame:
                    x = 0 if norm_dist == 0 else kp[0] / norm_dist
                    y = 0 if norm_dist == 0 else kp[1] / norm_dist
                    norm_kp.append([x, y])
                vid_skel.append(norm_kp)
            norm_skel.append(vid_skel)
        return norm_skel

    def _read_skeletons(self, max_samples=None):
        skeletons = []
        loop = tqdm(zip(os.listdir(self.label_dir), os.listdir(self.info_dir))) if max_samples is None else tqdm(zip(os.listdir(self.label_dir)[:max_samples], os.listdir(self.info_dir)[:max_samples]))
        i = 1
        labels = []
        for filename, info_name in loop:

            file_kp = self._read_file(self.label_dir + filename, self.info_dir + info_name)
            if file_kp is not None:
                labels.append(filename.split("-")[0])
                skeletons.append(file_kp)
            loop.set_postfix(filename=filename)
        return skeletons, labels

    
    def _read_file(self, filename, info_name):
        frames = []
        with open(info_name, "r") as info_file:
            info_text = info_file.read().strip()
            info_arr = info_text.split(";")
            info_dict = {}
            for inf in info_arr:
                k, v = inf.split(",")
                v = int(v)
                info_dict[k] = v

        with open(filename, "r") as kp_file:
            kp_txt = kp_file.read()
            if kp_txt == '':
                return None
            kp_arr = kp_txt.strip().split("\n")
            for frame in kp_arr:
                kps = frame.split(";")
                k_arr = []
                for k in kps:
                    points = k.split(",")
                    p_arr = []
                    for p in points:
                        curr_val = int(float(p))
                        self.C_min = min(curr_val, self.C_min)
                        self.C_max = max(curr_val, self.C_max)
                        p_arr.append(curr_val)
                    k_arr.append(p_arr)
                frames.append(global_coordinate_frame(k_arr))
            return frames


    def filter_peaks(self, peaks, threshold=5):
        prev_peak = 0
        new_peaks = []
        for p in peaks:
            if p - prev_peak > threshold:
                new_peaks.append(p)
            prev_peak = p
        return new_peaks

    def get_vid_peaks(self, norm_skel_vid):
        distances = []
        for i, kp in enumerate(norm_skel_vid):
            ankle1 = kp[mp_pose.PoseLandmark.LEFT_ANKLE]
            ankle2 = kp[mp_pose.PoseLandmark.RIGHT_ANKLE]
            distances.append(dist(ankle1, ankle2))
        filtered_distances = savgol_filter(distances, 9, 3)
        peaks = find_peaks(filtered_distances)[0]
        peaks = self.filter_peaks(peaks)
        return peaks, [filtered_distances[x] for x in peaks], filtered_distances

    def get_peaks(self, show_peaks=None):
        vid_peaks = []
        for i, v in enumerate(self.norm_skeletons):
            p, f, d = self.get_vid_peaks(v)
            if i == show_peaks:
                plt.figure()
                plt.plot(d)
                plt.scatter(p, f)
                plt.savefig("peaks.png")
                plt.close()
            vid_peaks.append(p)
        return vid_peaks


    def get_gait_cycles(self):
        """
        Gait cycles happen every second time ankle distances peak
        """
        peaks = self.get_peaks()
        g_arr = []
        for p in peaks:
            g_arr.append(p[::2])
        return g_arr

    def show_gait_cycle(self, vid_seq):
        peaks, peak_vals, distances = self.get_peaks(vid_seq)
        plt.plot(distances)
        plt.scatter(peaks, peak_vals)
        plt.savefig("output.png")
        plt.close()

class KinectDataset:
    def __init__(self, directory="KinectDataset/", max_samples=None):
        self.directory = directory
        self.skel_data, self.kp_indices = self._get_file_data(max_samples) # (n_people, n_files, n_lines, 3)
        
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
        self.pc = [(self.kp_indices[a], self.kp_indices[b]) for (a, b) in self.connections ]
        self.vid_skeletons, self.labels = self.reshape_skeletons()
        # quit()
        self.norm_skeletons = self.normalize_skeletons()
        
        self.vid_features = self.extract_features()

        self.gait_cycles = self.get_gait_cycles()
        self.gait_phases = self.get_gait_phases()
        self.compress_gait_phases()
        self.n_classes = len(np.unique(self.labels))
        self.shape = dim(self.gait_phases)

        # print(f"Gait Phases: {dim(self.gait_phases)}")
        # print(f"Labels: {len(self.labels)}")
        # quit()


    def _get_file_data(self, max_samples):
        kp_indices = {}
        ret_val = []
        if max_samples is None:
            loop = tqdm(os.listdir(self.directory))
        else:
            max_ind = int(len(os.listdir(self.directory)) * max_samples)
            loop = tqdm(os.listdir(self.directory)[:max_ind])
        for filename in loop:
            curr_person = []
            for f in os.listdir(self.directory + filename):
                curr_file = []
                with open(self.directory + filename + "/" +  f) as infile:
                    file_contents = infile.read().split('\n')
                    for i, x in enumerate(file_contents):
                        if x == '':
                            continue
                        z = float(x.split(';')[-1])
                        scale = 200 / (200 + z)
                        if x.split(';')[0] not in kp_indices.keys():
                            kp_indices[x.split(';')[0]] = i % 20
                        elif kp_indices[x.split(';')[0]] != i % 20:
                            print(f"Whoops {i} != {kp_indices[x.split(';')[0]]}")
                            quit()
                        curr_file.append([x.split(';')[0]] + [scale * float(a) for a in x.split(';')[1:3]])
                curr_person.append(curr_file)
            ret_val.append(curr_person)
            loop.set_postfix()
        return ret_val, kp_indices # (n_people, n_files, n_lines, 3)

    def reshape_skeletons(self):
        # Need skeleton to be shape (vid_seq, frame, keypoints, 2)
        labels = []
        reshaped_skel = []
        for i, person in enumerate(self.skel_data):
            # reshaped_skel.append([])
            for f in person:
                reshaped_skel.append([])
                labels.append(i)
                for l in f:
                    if l[0] == "Head":
                        reshaped_skel[-1].append([])
                    reshaped_skel[-1][-1].append([l[1], l[2]])
                if len(reshaped_skel[-1][-1]) != 20:
                    print(len(reshaped_skel[-1][-1]))
                    quit()
        return reshaped_skel, labels



    def normalize_skeletons(self):
        norm_skel = []
        for vid in self.vid_skeletons:
            vid_skel = []
            for frame in vid:
                point1 = frame[np.argmax(frame[:][1])]
                point2 = frame[np.argmin(frame[:][1])]
                norm_dist = dist(point1, point2)
                norm_kp = []
                for kp in frame:
                    x = 0 if norm_dist == 0 else kp[0] / norm_dist
                    y = 0 if norm_dist == 0 else kp[1] / norm_dist
                    norm_kp.append([x, y])
                vid_skel.append(norm_kp)
            norm_skel.append(vid_skel)
        return norm_skel

    def show_person(self, p=0, f=0):
        
        person = self.skel_data[p][f] # (n_lines, 3)
        assert person[0][0] == 'Head'
        person_skeleton = [person[0][:]]
        kp_dict = {"Head": person[0][1:]}
        for l in person[1:]: # l = (3,)
            if l[0] == 'Head':
                break
            person_skeleton.append(l[:])
            kp_dict[l[0]] = l[1:]


        plt.figure()
        plt.scatter([p[1] for p in person_skeleton], [p[2] for p in person_skeleton])
        for p in person_skeleton:
            plt.text(p[1], p[2], p[0])
        for from_val, to_val in self.connections:
            plt.plot(
                [kp_dict[from_val][0], kp_dict[to_val][0]],
                [kp_dict[from_val][1], kp_dict[to_val][1]])

        
        

        plt.savefig('output.png')
        plt.close()

    def filter_peaks(self, peaks, threshold=5):
        prev_peak = 0
        new_peaks = []
        for p in peaks:
            if p - prev_peak > threshold:
                new_peaks.append(p)
            prev_peak = p
        return new_peaks

    def get_vid_peaks(self, norm_skel_vid):
        distances = []
        for i, kp in enumerate(norm_skel_vid):
            ankle1 = kp[self.kp_indices["Ankle-Left"]]
            ankle2 = kp[self.kp_indices["Ankle-Right"]]
            distances.append(dist(ankle1, ankle2))
        filtered_distances = savgol_filter(distances, 9, 3)
        peaks = find_peaks(filtered_distances)[0]
        peaks = self.filter_peaks(peaks)
        return peaks, [filtered_distances[x] for x in peaks], filtered_distances

    def get_peaks(self, show_peaks=None):
        vid_peaks = []
        for i, v in enumerate(self.norm_skeletons):
            p, f, d = self.get_vid_peaks(v)
            if i == show_peaks:
                plt.figure()
                plt.plot(d)
                plt.scatter(p, f)
                plt.savefig("peaks.png")
                plt.close()
            vid_peaks.append(p)
        return vid_peaks

    def get_gait_cycles(self):
        """
        Gait cycles happen every second time ankle distances peak
        """
        peaks = self.get_peaks()
        g_arr = []
        for p in peaks:
            g_arr.append(p[::2])
        return g_arr

    def show_gait_cycle(self, vid_seq):
        peaks, peak_vals, distances = self.get_peaks(vid_seq)
        plt.plot(distances)
        plt.scatter(peaks, peak_vals)
        plt.savefig("output.png")
        plt.close()

    def compress_gait_phases(self):
        ret_val = []
        label_ret = []
        for sample, label in zip(self.gait_phases, self.labels):
            ret_val.extend(sample)
            label_ret.extend([label] * len(sample))
        self.labels = label_ret
        self.gait_phases = ret_val


    def extract_features(self):
        features = []
        total_skels = len(self.norm_skeletons)
        for i, vid in enumerate(self.norm_skeletons): 
            prev_skel = None
            frame_skels = []
            loop = tqdm(vid)
            for skel in loop:
                frame_skels.append(preprocess_kinect_dataset(skel, prev_skel, self.kp_indices, self.pc))
                prev_skel = skel
                loop.set_postfix(vid_number=f"{i+1}/{total_skels}")
            features.append(frame_skels)
        return features
    

    def get_gait_phases(self):
        gait_phases = [
            (0, 10),
            (10, 30),
            (30, 50),
            (50, 60),
            (60, 73),
            (73, 87),
            (87, 100)
        ]
        final_gait_features = []
        for i, v in enumerate(self.gait_cycles):
            vid_g_f = []
            if len(v) == 0:
                self.labels.pop(i)
                continue
            prev_g = v[0]
            for g in v[1:]:
                len_gait = g - prev_g
                curr_gp = []
                for gp in gait_phases:
                    start, stop = gp
                    start = prev_g + int(start / 100 * len_gait)
                    stop = prev_g + int(stop / 100 * len_gait)
                    curr_f = []
                    for f in range(5):
                        curr_vid = self.vid_features[i]
                        curr_frames = curr_vid[start:stop]
                        tmp = None
                        for fr in curr_frames:
                            tmp = fr[f] if tmp is None else [x + y for x, y in zip(fr[f], tmp)]
                        tmp = np.divide(tmp, stop - start)
                        curr_f.extend(tmp)
                    
                    curr_gp.append(curr_f)
                vid_g_f.append(curr_gp)
                prev_g = g
            final_gait_features.append(vid_g_f)
        return final_gait_features

    def show_gait_cycle(self, vid_seq=None):
        if vid_seq is None:
            vid_seq = self.vid_skeletons[0]
        self.get_peaks(1)

    def one_hot_labels(self):
        un_labels = list(np.unique(self.labels))
        self.n_classes = len(un_labels)
        onehot = np.zeros((len(self.labels), len(un_labels)))
        for i, l in enumerate(self.labels):
            l_ind = un_labels.index(l)
            onehot[i, l_ind] = 1
        return onehot