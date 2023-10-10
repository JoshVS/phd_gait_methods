import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from scipy.optimize import linear_sum_assignment as hungarian_algorithm_method
# from hungarian import hungarian_algorithm_method
import torch
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from types import SimpleNamespace

# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures
from sklearn.manifold import TSNE
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchviz import make_dot
def to_one_hot(y):
    n_c = len(np.unique(y))
    to_out = np.zeros((len(y), n_c))
    for i in range(to_out.shape[0]):
        to_out[i, y[i]] = 1
    return to_out

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x.float())
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # print(num_nodes)
        # quit()
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        # print(t2.shape)
        # quit()
        
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_classes):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_classes: Desired number of classes to be predicted.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        # print(dir(self.last_temporal))
        # quit()
        self.full1 = nn.Linear(90816, 128)#nn.Linear(33 * 47 * 64, 128)
        self.full2 = nn.Linear(128, 32)
        # print(self.full2.trace())
        # quit()
        self.fully = nn.Linear(32,
                               num_classes)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        

        # print(out3.shape)
        # quit()

        out4 = self.full1(torch.flatten(out3, start_dim=1, end_dim=3))
        out5 = self.full2(out4)

        # out4 = #.reshape((out3.shape[0], out3.shape[1], -1)))
        return F.softmax(self.fully(out5), dim=-1)
    

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


class GCNClassifier():
    """
    Classifier must take a dataset in initialiser

    Methods included:

    train(X, y)
    generate_test_set_results()
    predict(X)

    TODO: X must be of shape (n_samples, n_keypoints, 2)
    TODO: edges must be of shape (n_edges, 2)
    """
    def __init__(self, dataset, num_dims=3, num_phases=4):
        self.dataset = dataset
        self.X = torch.tensor(dataset.X_train).float()
        
        self.y = torch.tensor(to_one_hot(dataset.y_train))
        self.n_features = dim(dataset.X_train)[-1] // num_dims

        self.n_classes = dataset.n_classes
        self.edges = torch.tensor(dataset.edge_matrix)
        self.adj_mat = torch.tensor(self.dataset.create_graph_shift_operator()).float()
        print(self.X.shape)
        print(self.edges.shape)

        blocks = []
        blocks.append([self.X.shape[1]])
        blocks.append([64, 16, 64])
        blocks.append([64, 16, 64])
        blocks.append([self.X.shape[-1]])

       

        num_nodes = self.X.shape[2]
        # print(num_nodes, self.X.shape)
        # quit()
        num_features = self.X.shape[-1]
        num_timesteps = self.X.shape[2]

        self.classifier = STGCN(num_nodes, num_features, num_timesteps, self.n_classes)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.1)#, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train(self, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X
            y_train = self.y
        out = self.classifier(X_train.permute(0, 2, 1, 3), self.adj_mat)
        # print(out, y_train)
        # quit()
        # print(dir(out))
        # print(out.trace())
        # quit()
        make_dot(out).render("debugging", format="png")
        quit()
        loss = self.criterion(out, y_train)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def train(self, epochs, X_train=None, y_train=None):
        for epoch in range(1, epochs + 1):
            loss = self._train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    def test(self):
        self.model.eval()
        out = self.model(self.dataset.X_test, self.edges)
        pred = out.argmax(dim=1)
        test_correct = pred == self.dataset.y_test
        test_acc = int(test_correct.sum()) / len(self.dataset.y_train)
        return test_acc
    
    def generate_test_set_results(self):
        self.train(100)
        acc = self.test()
        print(f"Accuracy: {acc:.2f}")

    def predict(self, X):
        return self.classifier(X, self.edges)