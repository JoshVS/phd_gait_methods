"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import numpy as np
import torch
torch.set_default_dtype(torch.float32)


ul_keypoints_arr = [
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

ul_kp_indices = {}
for i, s in enumerate(ul_keypoints_arr):
    ul_kp_indices[s] = i

ul_connections = [
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

ul_edge_matrix = [[ul_kp_indices[x] for x, _ in ul_connections], [ul_kp_indices[y] for _, y in ul_connections]]
ul_in_edge = [
    (ul_kp_indices[x], ul_kp_indices[y])
    for (x, y) in ul_connections
]

# num_node = 59
# self_link = [(i, i) for i in range(num_node)]
# in_edge = [(0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17), (1, 2),
#            (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11),
#            (11, 12), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
#            (19, 20), (21, 22), (21, 26), (30, 34), (34, 38), (26, 30),
#            (21, 38), (22, 23), (23, 24), (24, 25), (26, 27), (27, 28),
#            (28, 29), (30, 31), (31, 32), (32, 33), (34, 35), (35, 36),
#            (36, 37), (38, 39), (39, 40), (40, 41), (42, 43), (43, 44),
#            (44, 45), (45, 49), (42, 46), (46, 47), (47, 48), (48, 50),
#            (51, 52), (53, 54), (53, 55), (55, 57), (54, 56), (56, 58)]
# out_edge = [(j, i) for (i, j) in in_edge]
# neighbor = in_edge + out_edge

def get_hop(link, num_node):
    """Calculates the identity matrix 

    Args:
        link (array of tuples): Edge connections between nodes
        num_node (int): Total number of nodes in the skeleton

    Returns:
        ndarray: Identity matrix
        For example: The indentity matrix of 3 nodes is 
        [[1,0,0],[0,1,0],[0,0,1]]
    """    
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    """Normalizes the graph value and returns the adjancy matrix

    Args:
        A (ndarray): Identity matrix (nodes, nodes)

    Returns:
        ndarray: Adjacency matrix (nodes, nodes)
    """    
    Dl = np.sum(A, 0) # Contains number of graphs
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, in_edge, out_edge):
    """_summary_

    Args:
        num_node (int): _description_
        self_link (list of tuple): links between self node. Node A has link to Node A.
        in_edge (list of tuple): links between inside edges
        out_edge (list of tuple): links between outside edge

    Returns:
        ndarray: Adjancy matrix containing information about the graph structure
    """    
    I = get_hop(self_link, num_node)    # For self links i.e., links between self node. Node A has link to Node A.
    In = normalize_digraph(get_hop(in_edge, num_node))  # For in_edge, i.e., links between inside edges
    Out = normalize_digraph(get_hop(out_edge, num_node))    # For out_edge, i.e., links between outside edge
    A = np.stack((I, In, Out), dtype=np.float32)  # Stack the three matrices together
    return A


class MediapipeGraph:
    def __init__(self, num_node, in_edge, labeling_mode='spatial'):
        self_link = [(i, i) for i in range(num_node)]
        out_edge = [(j, i) for (i, j) in in_edge]
        neighbor = in_edge + out_edge

        self.num_node = num_node
        self.self_link = self_link
        self.in_edge = in_edge
        self.out_edge = out_edge
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(self.num_node, self.self_link, self.in_edge, self.out_edge)
        else:
            raise ValueError()
        
        return A


def create_ultralytics_graph():
    
    torch.set_default_dtype(torch.float32)
    return MediapipeGraph(
        num_node=17,
        in_edge=ul_in_edge,
        labeling_mode='spatial'
    )