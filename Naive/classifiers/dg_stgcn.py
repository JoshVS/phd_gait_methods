import sys
import os
import glob
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from graphs.mpg import MediapipeGraph, create_ultralytics_graph
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from torchmetrics.functional import precision, recall
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from ray import train
from ray import tune
import ray
from ray.tune.tuner import Tuner
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile
import asyncio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system("rm -rf runs/*")
os.system("kill $(ps -e | grep 'tensorboard' | awk '{print $1}')")
async def start_tensorboard(direc):
    await asyncio.create_subprocess_shell("tensorboard --port 6006 --logdir=runs/" + direc)

device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_MODEL = 10
LOAD_MODEL = False
MODEL_NAME = "model_checkpoints"
TUNE = False
DROPOUT = 0.9
WEIGHT_DECAY = 1e-2
WEIGHTS_PATH = "robbery.pth"

if not os.path.exists(os.path.join("weights", "train", MODEL_NAME)):
    os.makedirs(os.path.join("weights", "train", MODEL_NAME))


if not os.path.exists(os.path.join("weights", "finished")):
    os.makedirs(os.path.join("weights", "finished"))

L1 = 3
L2 = 3
L3 = 3

BATCH_SIZE=32

EPOCHS = 10
LR = 1e-4

def force_cudnn_initialization():
    if device == "cuda":
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()


def weights_init(module_, bs=1):
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode='fan_out')
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(module_.weight, 0,
                        math.sqrt(2. / (module_.weight.size(0) * module_.weight.size(1) * module_.weight.size(2) * bs)))
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2. / bs))


# Helper function to normalize adjacency matrix (e.g., for graph convolution)
def normalize_adj(adj):
    """
    Normalizes the adjacency matrix for graph convolution.
    D^-1/2 * A * D^-1/2
    """
    degree_mat = torch.sum(adj, dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero for isolated nodes
    degree_mat_inv_sqrt = torch.pow(degree_mat + 1e-12, -0.5)
    degree_mat_inv_sqrt[degree_mat_inv_sqrt == float('inf')] = 0.0 # Handle isolated nodes
    adj_normalized = degree_mat_inv_sqrt * adj * degree_mat_inv_sqrt.transpose(-1, -2)
    return adj_normalized

class DynamicGraphConv(nn.Module):
    """
    A conceptual Dynamic Graph Convolutional Layer.
    It learns an adjacency matrix based on input features and then applies
    a graph convolution using this learned matrix.
    """
    def __init__(self, in_channels, out_channels, num_nodes):
        super(DynamicGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes # This is V_joints * M_persons (total keypoints)

        # Learnable parameters for generating the dynamic adjacency matrix
        # A simple linear layer to project features into an affinity space.
        # This can be more complex (e.g., multiple layers, attention mechanisms)
        self.affinity_learner = nn.Linear(in_channels, num_nodes)

        # Weight matrix for the graph convolution operation
        self.conv_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.kaiming_uniform_(self.conv_weight, a=0.01) # Initialize weights

        # Bias for the convolution
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        x: Input feature tensor of shape (batch_size, in_channels, num_nodes)
        """
        batch_size = x.size(0)

        # 1. Learn Dynamic Adjacency Matrix
        # Transpose x to (batch_size, num_nodes, in_channels) for linear layer
        x_for_affinity = x.permute(0, 2, 1)
        
        # Generate a "base" affinity matrix from features.
        # This is a simple projection. A more sophisticated approach would use
        # a self-attention mechanism (e.g., dot product of queries and keys).
        # Output shape: (batch_size, num_nodes, num_nodes)
        learned_affinity = self.affinity_learner(x_for_affinity)
        
        # Apply softmax or sigmoid to make it a valid adjacency matrix (e.g., probabilities)
        # Using softmax along the last dimension to ensure rows sum to 1,
        # or sigmoid for independent edge probabilities.
        # For a truly symmetric graph, you might average (learned_affinity + learned_affinity.transpose(-1, -2))
        adj = F.softmax(learned_affinity, dim=-1)
        
        # You might want to add the original fixed skeleton graph here as a residual connection
        # For simplicity, we are only using the learned dynamic graph.

        # 2. Normalize Adjacency Matrix
        adj_normalized = normalize_adj(adj)

        # 3. Apply Graph Convolution
        # Reshape x to (batch_size, num_nodes, in_channels) for matrix multiplication
        x_reshaped = x.permute(0, 2, 1)

        # Graph convolution: (A * X) * W
        # (batch_size, num_nodes, num_nodes) @ (batch_size, num_nodes, in_channels)
        # -> (batch_size, num_nodes, in_channels)
        graph_output = torch.bmm(adj_normalized, x_reshaped)

        # Apply convolution weight: (batch_size, num_nodes, in_channels) @ (in_channels, out_channels)
        # -> (batch_size, num_nodes, out_channels)
        output = torch.matmul(graph_output, self.conv_weight)

        # Add bias and reshape back to (batch_size, out_channels, num_nodes)
        output = output + self.bias
        output = output.permute(0, 2, 1) # Back to (batch_size, out_channels, num_nodes)

        return output, adj # Return output features and the learned adjacency matrix


class DGSTGCNBlock(nn.Module):
    """
    A conceptual block combining Dynamic Spatial Graph Convolution and Temporal Convolution.
    Inspired by ST-GCN block structure.
    """
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=(9, 1), dropout=0.5):
        super(DGSTGCNBlock, self).__init__()
        # kernel_size: (temporal_kernel_size, spatial_kernel_size)
        # Spatial kernel size is implicitly handled by the graph structure.

        self.dg_conv = DynamicGraphConv(in_channels, out_channels, num_nodes)

        # Temporal convolution (1D convolution along the time dimension)
        # Input to temporal conv: (batch_size, out_channels, num_frames, num_nodes)
        # We need to reshape the output of DG-Conv to include time dimension for this.
        # For simplicity here, we assume DG-Conv operates on a single frame's data,
        # and then temporal conv operates across frames.
        # In a full ST-GCN, the spatial and temporal convolutions are interleaved.
        
        # For this conceptual block, let's assume the input to the block already has
        # a temporal dimension, and we apply DG-Conv per frame, then temporal conv.
        # Or, more simply, DG-Conv handles the spatial part, and a separate conv handles temporal.
        
        # Let's define it as a standard temporal convolution that will operate
        # after the spatial graph convolution.
        # It expects input (N, C, T, V), where V is num_nodes here.
        # We'll adapt the DG-Conv output for this.
        
        # The kernel_size[0] is the temporal kernel size.
        # The padding ensures the output sequence length is the same as input.
        pad = (kernel_size[0] - 1) // 2
        self.tcn = nn.Conv1d(out_channels, out_channels, kernel_size[0], padding=pad)
        
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(out_channels) # BatchNorm after convolution
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = lambda x: x

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, in_channels, num_frames, num_nodes)
        """
        N, C, T, V = x.size()
        
        # Apply residual connection
        # Reshape for 1D conv: (N * V, C, T)
        res = self.residual(x.reshape(N * V, C, T))
        
        # Reshape for DynamicGraphConv: (N * T, C, V)
        # We apply DG-Conv to each frame independently.
        x_reshaped_for_dg = x.permute(0, 2, 1, 3).reshape(N * T, C, V)
        
        # Apply Dynamic Graph Convolution
        # Output: (N * T, out_channels, V)
        spatial_features, learned_adj = self.dg_conv(x_reshaped_for_dg)
        
        # Reshape back for temporal convolution: (N, out_channels, T, V)
        # Then flatten for 1D temporal conv: (N * V, out_channels, T)
        spatial_features_reshaped = spatial_features.view(N, T, self.dg_conv.out_channels, V)
        spatial_features_reshaped = spatial_features_reshaped.permute(0, 3, 2, 1).reshape(N * V, self.dg_conv.out_channels, T)

        # Apply Temporal Convolution
        # Output: (N * V, out_channels, T)
        temporal_features = self.tcn(spatial_features_reshaped)
        
        # Add residual connection
        output = temporal_features + res
        
        # Apply BatchNorm and ReLU
        output = self.bn(output)
        output = self.relu(output)
        
        # Reshape back to (N, out_channels, T, V)
        output = output.view(N, V, self.dg_conv.out_channels, T).permute(0, 2, 3, 1)

        return output, learned_adj # Return processed features and the last learned adjacency matrix

class GeminiDGSTGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, graph, num_timestep, cuda_=torch.cuda.is_available(), l1=L1, l2=L2, l3=L3, dropout=DROPOUT):
        super(GeminiDGSTGCN, self).__init__()

        self.graph = graph

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        weights_init(self.data_bn, bs=1)
        layers = [DGSTGCNBlock(in_channels, 64, num_point,  dropout=dropout)]
        layers += [DGSTGCNBlock(64, 64, num_point,  dropout=dropout)] * (l1 - 1)
        layers += [DGSTGCNBlock(64, 128, num_point, dropout=dropout)]
        layers += [DGSTGCNBlock(128, 128, num_point,  dropout=dropout)] * (l2 - 1)
        layers += [DGSTGCNBlock(128, 256, num_point, dropout=dropout)]
        layers += [DGSTGCNBlock(256, 256, num_point,  dropout=dropout)] * (l3 - 1)
        
        # print(layers)
        # quit()
        # layers = [ 
        #     ST_GCN_block(in_channels, 64, A, cuda_, residual=False),
        #     ST_GCN_block(64, 64, A, cuda_),
        #     #  'layer3': ST_GCN_block(64, 64, A, cuda_),
        #     ST_GCN_block(64, 64, A, cuda_),
        #     ST_GCN_block(64, 128, A, cuda_, stride=2),
        #     ST_GCN_block(128, 128, A, cuda_),
        #     ST_GCN_block(128, 128, A, cuda_),
        #     ST_GCN_block(128, 256, A, cuda_, stride=2),
        #     ST_GCN_block(256, 256, A, cuda_),
        #     ST_GCN_block(256, 256, A, cuda_)
        # ]
        layer_dict = {}
        for i, l in enumerate(layers):
            layer_dict[f'layer{i+1}'] = l

        self.layers = nn.ModuleDict(layer_dict)

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(self, x):
        """
        X: Shape (batch, channels, time, nodes, people)
        """
        N, C, T, V, M = x.size()

        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (batch, people * nodes * channels, times)
        
        x = self.data_bn(x) # batchnorm
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) # (batch * people, channels, times, nodes)
        for i in range(len(self.layers)):
            x,_ = self.layers['layer' + str(i+1)](x)
        # N*M,C,T,V

        c_new = x.size(1) # infer new channel size
        x = x.reshape(N, M, c_new, -1) # (batch, people, new_channel_size, times * nodes)
        x = x.mean(3).mean(1) # Take mean across times*nodes and people
        # return softmax(self.fc(x), dim=1) # in shape: (batch, new_channel_size)
        return softmax(self.fc(x), dim=-1) # in shape: (batch, times, num_classes)

class DGSTGCN:
    def __init__(self, ds=None, loss_fn=torch.nn.functional.cross_entropy, model_name=MODEL_NAME, weights_path=None, timesteps=None, testing=False):
        
        torch.set_default_dtype(torch.float32)
        self.model_name = model_name

        if weights_path is not None and testing:
            self.graph = create_ultralytics_graph()
            model_weights = torch.load(weights_path, map_location=device )
            self.classifier = GeminiDGSTGCN(2,17, 1, 2, self.graph, timesteps).to(device)
            self.classifier.load_state_dict(model_weights)
            print(f"Loaded model from {weights_path}")
            scalar_metrics = {"train":{},
                            "val":{}}
            
            test_metrics = {"scalar": {}, "image": {}}
            # print(self.test_set.X[0].size())
            # quit()
            test_iter = iter(self.test_set)
            for idx, test_sample in enumerate(test_iter):
                # print(test_sample[0].size())
                # quit()
                test_metrics = self._test_step(test_sample, test_metrics)
                for k in test_metrics["scalar"].keys():
                    if k not in scalar_metrics["test"].keys():
                        scalar_metrics["test"][k] = 0
                    scalar_metrics["test"][k] += test_metrics["scalar"][k] / len(self.test_set.X)

            return

        if weights_path is not None:
            self.graph = create_ultralytics_graph()
            model_weights = torch.load(weights_path, map_location=device )
            self.classifier = GeminiDGSTGCN(2,17, 1, 2, self.graph, timesteps).to(device)
            self.classifier.load_state_dict(model_weights)
            print(f"Loaded model from {weights_path}")
            return

        self.train_set, self.val_set = ds
        # print(len(self.train_set)//866)
        # quit()
        ds = self.train_set
        self.time_steps = ds.num_timesteps
        self.n_classes = ds.n_classes
        self.class_names = ds.classes
        self.n_point = ds.n_point
        self.num_person = ds.num_person
        self.in_channels = ds.in_channels
        self.loss_fn = loss_fn

        self.graph = MediapipeGraph(self.n_point, ds.in_edge)

        # self.classifier = GeminiDGSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph).to(device)
        # if not os.path.exists(model_name):
        #     os.makedirs(model_name)
        if LOAD_MODEL:
            if not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}")):
                print("No models found, creating a new one")
            elif not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}")):
                print("No models found, creating a new one")
            else:
                model_names = os.listdir(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}"))
                if len(model_names) == 0:
                    print("No models found, creating a new one")
                else:
                    model_files = os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}", model_names[-1])#os.path.join(model_name, model_names[-1])
                    print(f"Loading model from {model_files}")
                    self.classifier.load_state_dict(torch.load(model_files))

        self.tracking_metrics = {
            # "precision": lambda x,y: precision(x, y, 'multilabel', num_classes=self.n_classes),
            # "recall": lambda x,y: recall(x, y, 'multilabel', num_classes=self.n_classes),
            # "accuracy": Accuracy("multiclass", average="macro", num_classes=self.n_classes).to(device),
            # "precision": Precision("multiclass", average="macro", num_classes=self.n_classes).to(device),
            # "recall": Recall("multiclass", average="macro", num_classes=self.n_classes).to(device)
            "accuracy": lambda x, y: accuracy_score(x.cpu().numpy(), y.cpu().numpy()),
            "precision": lambda x, y: precision_score(x.cpu().numpy(), y.cpu().numpy(), average="macro", zero_division=1.0),
            "recall": lambda x, y: recall_score(x.cpu().numpy(), y.cpu().numpy(), average="macro", zero_division=1.0)
        }
        self.train()
        # print(train_data.dtype)
        # quit()

    def predict(self, sample):
        return self.classifier(sample.to(device))
        

    def _train_step(self, sample, optimizer, metrics, val_sample=None, classifier=None):
        if classifier is None:
            classifier = self.classifier
        metrics["scalar"] = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device).view(-1, self.n_classes).float()  # Flatten the labels to match the output shape of the classifier
        # print(y.size())
        optimizer.zero_grad()
        outputs = classifier(X).float()
        outputs = outputs.view(-1, self.n_classes)  # Flatten the outputs to match the output shape of the classifier
        predictions = outputs.argmax(axis=1)
        true_labels = y.argmax(axis=1)
        # predictions = predictions.view(-1, predictions.size(2))  # Flatten the predictions to match the output shape of the classifier
        loss = self.loss_fn(outputs, y)

        loss.backward()

        optimizer.step()
        metrics["scalar"]['loss'] = loss.item()
        with torch.no_grad():
            for k in self.tracking_metrics.keys():
                metrics["scalar"][k] = self.tracking_metrics[k](true_labels, predictions)
            # print(y.cpu().numpy().argmax(axis=1).shape, predictions.cpu().numpy().argmax(axis=1).shape)
            # quit()
            cm = confusion_matrix(true_labels.cpu().numpy(), predictions.cpu().numpy(), labels = np.array(list(range(self.n_classes))))
            if "conf_mat" not in metrics["image"].keys():
                # print(cm.shape)
                metrics["image"]["conf_mat"] = cm

            else:
                # print(cm.shape)
                metrics["image"]["conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm
                    
        return metrics

    def _val_step(self, sample, val_metrics, classifier=None):
        if classifier is None:
            classifier = self.classifier
        val_metrics["scalar"] = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device).view(-1, self.n_classes).float()  # Flatten the labels to match the output shape of the classifier
        
        with torch.no_grad():
            val_out = classifier(X).view(-1, self.n_classes).float()
            predictions = val_out.argmax(axis=1)
            true_labels = y.argmax(axis=1)
            # predictions = predictions.view(-1, predictions.size(2))  # Flatten the predictions to match the output shape of the classifier
            # print(y.dtype)
            # quit()
            val_loss = self.loss_fn(val_out, y)
            val_metrics["scalar"]["val_loss"] = val_loss.item()
            for k in self.tracking_metrics.keys():
                val_metrics["scalar"]["val_" + k] = self.tracking_metrics[k](true_labels, predictions)

            cm = confusion_matrix(true_labels.cpu().numpy(), predictions.cpu().numpy(), labels = np.array(list(range(self.n_classes))))
            if "val_conf_mat" not in val_metrics["image"]:
                # print(cm.shape)
                val_metrics["image"]["val_conf_mat"]  = cm

            else:
                # print(cm.shape)
                # print(val_metrics["image"]["val_conf_mat"].shape, cm.shape)
                val_metrics["image"]["val_conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm

        
        return val_metrics

    def _test_step(self, sample, test_metrics, classifier=None):
        if classifier is None:
            classifier = self.classifier
        test_metrics["scalar"] = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device).view(-1, self.n_classes).float()  # Flatten the labels to match the output shape of the classifier
        
        with torch.no_grad():
            test_out = classifier(X).view(-1, self.n_classes).float()
            predictions = test_out.argmax(axis=1)
            true_labels = y.argmax(axis=1)
            # predictions = predictions.view(-1, predictions.size(2))  # Flatten the predictions to match the output shape of the classifier
            # print(y.dtype)
            # quit()
            test_loss = self.loss_fn(test_out, y)
            test_metrics["scalar"]["test_loss"] = test_loss.item()
            for k in self.tracking_metrics.keys():
                test_metrics["scalar"]["test_" + k] = self.tracking_metrics[k](true_labels, predictions)

            cm = confusion_matrix(true_labels.cpu().numpy(), predictions.cpu().numpy(), labels = np.array(list(range(self.n_classes))))
            if "test_conf_mat" not in test_metrics["image"]:
                # print(cm.shape)
                test_metrics["image"]["test_conf_mat"]  = cm

            else:
                # print(cm.shape)
                # print(test_metrics["image"]["test_conf_mat"].shape, cm.shape)
                test_metrics["image"]["test_conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm

        
        return test_metrics

    def train(self,  lr=LR, momentum=0.9, epochs=EPOCHS, batch_size=BATCH_SIZE):
        
        self.train_set.to(device)
        self.val_set.to(device)

        t = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[2] / 2) / 2) / 2)
        h = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[3] / 2) / 2) / 2) // 2
        w = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[4] / 2) / 2) / 2)
        # print((t,h,w))
        # quit()

        # print((self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, self.time_steps))
        # quit()

        self.classifier = GeminiDGSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, self.time_steps, dropout=DROPOUT).to(device)        
        param_size = 0
        for param in self.classifier.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.classifier.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {total_size:.2f} MB")
        
        # def ray_train_func(train_iter, val_set, optimizer, train_metrics, val_metrics): 
                
            # for idx, val_sample in enumerate(val_set):
            #     val_metrics = self._val_step(val_sample, val_metrics)
        
        # @ray.remote
        def tune_hyperparams(config, data=None):
            classifier = ray.get(self.classifier)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir)
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    classifier.load_state_dict(checkpoint_state["net_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                start_epoch = 0
            
            # if device == "cuda":
            #     classifier = nn.DataParallel(classifier, device_ids=[0], output_device=0)
            classifier.to(device)
                       
           
            train_set, val_set = data
            for epoch in range(start_epoch, start_epoch + epochs):
                print()
                print(f"Epoch #{epoch + 1}: {self.num_timesteps} Timesteps")
                
                train_metrics = {"scalar": {}, "image": {}}
                val_metrics = {"scalar": {}, "image": {}}
                
                for idx, curr_sample in enumerate(train_set):
                    cs = ray.get(curr_sample)
                    train_metrics = self._train_step(cs, optimizer, train_metrics, classifier=classifier)
                    del cs
                for idx, val_sample in enumerate(val_set):
                    vs = ray.get(val_sample)
                    val_metrics = self._val_step(vs, val_metrics, classifier=classifier)
                    del vs
        
                checkpoint_data = {
                    "epoch":epoch,
                    "net_state_dict": classifier.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict()
                    
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    # print(val_metrics)
                    tune.report(val_metrics['scalar'], checkpoint=checkpoint)

        if TUNE:
            ray.init(num_cpus=4, num_gpus=1, include_dashboard=True)
            self.classifier = ray.put(self.classifier)
            config = {
                "lr": tune.loguniform(1e-5, 1e-1),
                "dropout": tune.loguniform(1e-1, 0.9),
                "weight_decay": tune.loguniform(1e-5, 1e-1),
            }
            # test_config = {
            #     "lr": 1e-5,
            #     "dropout": 0.25,
            #     "weight_decay": 1e-5
            # }
            train_ray = [None] * self.train_set.num_batches
            for i in range(self.train_set.num_batches):
                train_ray[i] = ray.put(self.train_set[i])

            val_ray = [None] * self.val_set.num_batches
            for i in range(self.val_set.num_batches):
                val_ray[i] = ray.put(self.val_set[i])

            # tune_hyperparams(config=test_config, data=(train_ray, val_ray))
            # quit()


            tuner = Tuner(
                trainable=tune.with_parameters(tune.with_resources(tune_hyperparams, {"gpu": 1}), data=(train_ray, val_ray)),
                param_space=config,
                tune_config=tune.TuneConfig(
                    metric="val_loss",
                    mode="min",
                    num_samples=30,
                    scheduler=tune.schedulers.ASHAScheduler(time_attr='epoch', max_t=30)
                )
                # run_config=tune.RunConfig(
                #     name="tune_hyperparams",
                #     # storage_path="C:\\\\Users\\joshua.vanstaden\\Documents\\Models\\phd_gait_methods\\Naive",
                # )
            )
            result = tuner.fit()

            # result = tune.run(
            #     partial(tune_hyperparams, data_dir="tuning"),
            #     resources_per_trial={"cpu": 1, "gpu": 1},
            #     config=config,
            #     scheduler=scheduler
            # )
            print(dir(result.get_best_result(metric="val_loss", mode="min")))
            best_trial = result.get_best_result()
            print(f"Best trial config: \t {best_trial.config}")
            print(f"Best Trial Final Validation Metrics: \t {best_trial.metrics_dataframe}")

            best_trained_model = GeminiDGSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, self.time_steps, dropout=best_trial.config["dropout"]).to(device)

            best_checkpoint = best_trial.get_best_checkpoint(metric="val_accuracy", mode="max")

            with best_checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    best_checkpoint_data = pickle.load(fp)

                best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        else:
            writer = SummaryWriter()
            asyncio.run(start_tensorboard(os.listdir("runs")[-1]))
          
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            for epoch in range(0, epochs):
                # print(self.val_set.ds.X.size())
                # quit()

                print()
                print(f"Epoch #{epoch + 1}: ")
                scalar_metrics = {"train":{},
                                "val":{}}
                
                val_metrics = {"scalar": {}, "image": {}}
                # print(self.val_set.X[0].size())
                # quit()
                val_iter = iter(self.val_set)
                for idx, val_sample in enumerate(val_iter):
                    # print(val_sample[0].size())
                    # quit()
                    val_metrics = self._val_step(val_sample, val_metrics)
                    for k in val_metrics["scalar"].keys():
                        if k not in scalar_metrics["val"].keys():
                            scalar_metrics["val"][k] = 0
                        scalar_metrics["val"][k] += val_metrics["scalar"][k] / len(self.val_set.X)
                train_iter = iter(self.train_set)
                print("##################")
                for k in scalar_metrics["val"].keys():
                
                    print(f"{k.capitalize()}: {scalar_metrics['val'][k]:.3f}")
                print("##################")
                writer.add_scalars("Validation", scalar_metrics["val"], epoch)
                # print("GOT HERE")
                # quit()

                loop = tqdm(enumerate(train_iter))
                train_metrics = {"scalar": {}, "image": {}}
                for idx, curr_sample in loop:
                    train_metrics = self._train_step(curr_sample, optimizer, train_metrics)
                    for k in train_metrics["scalar"].keys():
                        if k not in scalar_metrics["train"].keys():
                            scalar_metrics["train"][k] = 0
                        
                        scalar_metrics["train"][k] += train_metrics["scalar"][k] / len(self.train_set.X)
                    # train.report(scalar_metrics["train"])
                    loop.set_postfix(train_metrics["scalar"])
                writer.add_scalars("Training", scalar_metrics["train"], epoch)
                train_hm = train_metrics["image"]["conf_mat"]
                # sum = 0
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         sum += train_hm[i][j]
                
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         train_hm[i][j] /= sum

                sns.heatmap(train_hm, annot=True, xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                writer.add_figure("Training Confusion Matrix", plt.gcf(), epoch)
                plt.close()

                train_hm = val_metrics["image"]["val_conf_mat"]
                # sum = 0
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         sum += train_hm[i][j]
                
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         print(f"{train_hm[i][j]} / {sum} = {train_hm[i][j] / sum}")
                #         train_hm[i][j] /= sum

                
                sns.heatmap(train_hm, annot=True, xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                writer.add_figure("Validation Confusion Matrix", plt.gcf(), epoch)

                if SAVE_MODEL is not None:
                    if epoch % SAVE_MODEL == 0:
                        torch.save(self.classifier.state_dict(), os.path.join("weights", "train", self.model_name, f"valloss_{scalar_metrics['val']['val_loss']}_timesteps_{self.time_steps}_classes_{self.n_classes}_model_epoch_{epoch}.pth"))
            torch.save(self.classifier.state_dict(), os.path.join("weights", "finished", WEIGHTS_PATH))

# --- Example Usage ---
if __name__ == '__main__':
    # Dummy input parameters
    batch_size = 2
    in_channels = 3  # e.g., (x, y, confidence) for each joint
    num_frames = 30  # Number of frames in the video clip
    num_joints_per_person = 17 # e.g., COCO keypoints
    max_persons = 2 # Maximum number of people to track

    # Total number of nodes in the graph (joints from all people)
    total_graph_nodes = num_joints_per_person * max_persons

    # Input tensor shape: (batch_size, in_channels, num_frames, total_graph_nodes)
    # This assumes keypoints for all people are concatenated along the last dimension.
    # For example, if person 1 has joints 0-16 and person 2 has joints 17-33.
    dummy_input = torch.randn(batch_size, in_channels, num_frames, total_graph_nodes)

    print(f"Dummy Input Shape: {dummy_input.shape}")

    # Initialize the DGSTGCNBlock
    out_channels = 64 # Output channels after the block
    dg_stgcn_block = DGSTGCNBlock(in_channels, out_channels, total_graph_nodes)

    # Forward pass
    output_features, learned_adjacency_matrix = dg_stgcn_block(dummy_input)

    print(f"Output Features Shape: {output_features.shape}")
    # The learned adjacency matrix is for a single frame (N*T, V, V)
    # Here we show the shape for the last frame of the last batch.
    print(f"Learned Adjacency Matrix Shape (for one frame): {learned_adjacency_matrix.shape}")

    # You can inspect the learned adjacency matrix for a specific frame/batch
    # For example, to see the adjacency matrix for the first batch, first frame:
    # (Note: learned_adjacency_matrix contains matrices for all N*T frames)
    print("\nExample Learned Adjacency Matrix (first batch, first frame):")
    print(learned_adjacency_matrix[0].detach().numpy())