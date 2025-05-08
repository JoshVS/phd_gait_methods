import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

import os
import sys
from collections import OrderedDict
from ray import tune


import sys
import os
import glob
from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graphs.mpg import MediapipeGraph
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
from ray.tune.tuner import Tuner
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_MODEL = 1
LOAD_MODEL = True
MODEL_NAME = "model_checkpoints"
TUNE = False
DROPOUT = 0.75
WEIGHT_DECAY = 1e-4

BATCH_SIZE=8

EPOCHS = 100
LR = 1e-4



class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        # print(x.size())
        # quit()
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionI3dGraph(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_class, num_point, num_person, in_channels, graph, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d',  dropout_keep_prob=0.5, l1=1, l2=1, l3=1,
                 dropout=0.5, thw=(2,2,7)):
        
        # num_class, num_point, num_person, in_channels, graph
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3dGraph, self).__init__()
        t, h, w = thw
        self._num_classes = num_class
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3,3,3),  name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                       name=name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64,96,128,16,32,32], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128,128,192,32,96,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128+192+96+64, [192,96,208,16,48,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192+208+48+64, [160,112,224,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160+224+64+64, [128,128,256,24,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128+256+64+64, [112,144,288,32,64,64], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112+288+64+64, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                             padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [256,160,320,32,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256+320+128+128, [384,192,384,48,128,128], name+end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[t, h, w],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:

                x = self._modules[end_point](x) # use _modules to work with dataparallel
                # quit()
                # print(x.size())
        # quit()
        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return logits[...,0]
        

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
    



class InceptionClassifier:
    def __init__(self, ds, loss_fn=torch.nn.functional.cross_entropy, model_name=MODEL_NAME):
        
        torch.set_default_dtype(torch.double)
        self.model_name = model_name
        self.train_set, self.test_set, self.val_set = ds
        # print(len(self.train_set)//866)
        # quit()
        ds = self.train_set
        self.time_steps = ds.num_timesteps
        self.n_classes = ds.n_classes
        self.class_names = ds.classes
        self.n_point = ds.n_point
        self.num_person = 1
        self.in_channels = ds.in_channels
        self.loss_fn = loss_fn

        self.graph = MediapipeGraph(self.n_point, ds.in_edge)

        # self.classifier = InceptionI3dGraph(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph).to(device)
        # if not os.path.exists(model_name):
        #     os.makedirs(model_name)
        # if LOAD_MODEL:
        #     if not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}")):
        #         print("No models found, creating a new one")
        #     elif not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}")):
        #         print("No models found, creating a new one")
        #     else:
        #         model_names = os.listdir(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}"))
        #         if len(model_names) == 0:
        #             print("No models found, creating a new one")
        #         else:
        #             model_files = os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}", model_names[-1])#os.path.join(model_name, model_names[-1])
        #             print(f"Loading model from {model_files}")
        #             self.classifier.load_state_dict(torch.load(model_files))

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

        

    def _train_step(self, sample, optimizer, metrics, val_sample=None):
        metrics["scalar"] = {}
        if val_sample is not None:
            val_metrics = {}
        val_metrics = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device)
        # print(y.size())
        optimizer.zero_grad()
        outputs = self.classifier(X)
        predictions = torch.nn.functional.one_hot(outputs.argmax(axis=1), num_classes=self.n_classes)
        loss = self.loss_fn(outputs, y)

        loss.backward()

        optimizer.step()
        metrics["scalar"]['loss'] = loss.item()
        with torch.no_grad():
            for k in self.tracking_metrics.keys():
                metrics["scalar"][k] = self.tracking_metrics[k](y, predictions)
            # print(y.cpu().numpy().argmax(axis=1).shape, predictions.cpu().numpy().argmax(axis=1).shape)
            # quit()
            # cm = confusion_matrix(y.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1), labels = np.array(list(range(self.n_classes))))
            # if "conf_mat" not in metrics["image"].keys():
            #     # print(cm.shape)
            #     metrics["image"]["conf_mat"] = cm

            # else:
            #     # print(cm.shape)
            #     metrics["image"]["conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm
                    
        return metrics

    def _val_step(self, sample, val_metrics):
        val_metrics["scalar"] = {}
        X, y = sample
        # print(X.size())
        # quit()
        X = X.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            # print(X.size())
            # quit()
            val_out = self.classifier(X)
            predictions = torch.nn.functional.one_hot(val_out.argmax(axis=1), num_classes=self.n_classes)
            # print(val_out.size(), y.size())
            val_loss = self.loss_fn(val_out, y)
            val_metrics["scalar"]["val_loss"] = val_loss.item()
            for k in self.tracking_metrics.keys():
                val_metrics["scalar"]["val_" + k] = self.tracking_metrics[k](y, predictions)
              
            # cm = confusion_matrix(y.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1), labels = np.array(list(range(self.n_classes))))
            # if "val_conf_mat" not in val_metrics["image"]:
            #     # print(cm.shape)
            #     val_metrics["image"]["val_conf_mat"]  = cm

            # else:
            #     # print(cm.shape)
            #     # print(val_metrics["image"]["val_conf_mat"].shape, cm.shape)
            #     val_metrics["image"]["val_conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm

        

        return val_metrics


    def train(self,  lr=LR, momentum=0.9, epochs=EPOCHS, batch_size=BATCH_SIZE):
        
        self.total_train_set = DataLoader(self.train_set, batch_size=batch_size, shuffle=False)
        self.total_val_set = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        t = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[2] / 2) / 2) / 2)
        h = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[3] / 2) / 2) / 2) // 2
        w = math.ceil(math.ceil(math.ceil(self.train_set.ds.X.size()[4] / 2) / 2) / 2)
        # print((t,h,w))
        # quit()

        self.classifier = InceptionI3dGraph(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, l1=3, l2=3, l3=3, dropout=0.5, thw=(t, h, w)).to(device)        
        param_size = 0
        for param in self.classifier.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.classifier.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = (param_size + buffer_size) / 1024**2
        print(f"Model Size: {total_size:.2f} MB")
        
        

        def tune_hyperparams(config, data=None):
            self.train_set, self.val_set = data
            # num_class, num_point, num_person, in_channels, graph
            # self.classifier = InceptionI3dGraph(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, l1=config["l1"], l2=config["l2"], l3=config["l3"], dropout=config["dropout"])
            
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=config['lr'], weight_decay=WEIGHT_DECAY)
            checkpoint = get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir)
                    with open(data_path, "rb") as fp:
                        checkpoint_state = pickle.load(fp)
                    start_epoch = checkpoint_state["epoch"]
                    self.classifier.load_state_dict(checkpoint_state["net_state_dict"])
                    optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
            else:
                start_epoch = 0
            
            if device == "cuda":
                self.classifier = nn.DataParallel(self.classifier, device_ids=[0], output_device=0)
            self.classifier.to(device)

        # train_samples = int((1 - test_split) * len(self.ds))
        # test_samples = int(len(self.ds) - train_samples)
        # val_samples = int(val_split * train_samples)
        # train_samples = int(train_samples - val_samples)
        # self.train_set, self.test_set, self.val_set = torch.utils.data.random_split(self.ds, [train_samples, test_samples, val_samples])
        # self.val_set = self.val_set.to(device)

        # quit()
        # print(len(self.train_set))
        # quit()


            # optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

            # writer = SummaryWriter()

        # for epoch in range(epochs):
        #     train_iter = iter(self.train_set)
        #     curr_sample = next(train_iter)
        #     while curr_sample is not None:
        #         self._train_step(curr_sample, optimizer)
        #         curr_sample = next(train_iter)
        

            # print(f"Training with {self.time_steps} time steps and {self.n_classes} classes")
            # print(f"Training on device {device}")
            for epoch in range(start_epoch, start_epoch + epochs):
                print()
                print(f"Epoch #{epoch + 1}: ")
                scalar_metrics = {"train":{},
                                "val":{}}
                
                val_metrics = {"scalar": {}, "image": {}}
                val_iter = iter(self.val_set)
                for idx, val_sample in enumerate(val_iter):
                    val_metrics = self._val_step(val_sample, val_metrics)
                    for k in val_metrics["scalar"].keys():
                        if k not in scalar_metrics["val"].keys():
                            scalar_metrics["val"][k] = 0
                        scalar_metrics["val"][k] += val_metrics["scalar"][k] / len(self.val_set)
                train_iter = iter(self.train_set)

                loop = tqdm(enumerate(train_iter))
                train_metrics = {"scalar": {}, "image": {}}
                for idx, curr_sample in loop:
                    train_metrics = self._train_step(curr_sample, optimizer, train_metrics)
                    for k in train_metrics["scalar"].keys():
                        if k not in scalar_metrics["train"].keys():
                            scalar_metrics["train"][k] = 0
                        
                        scalar_metrics["train"][k] += train_metrics["scalar"][k] / len(self.train_set)
                    # train.report(scalar_metrics["train"])
                    loop.set_postfix(train_metrics["scalar"])
                    # train_iter.set_postfix(train_metrics["scalar"])
                # sns.heatmap(train_metrics["image"]["conf_mat"], annot=False, xticklabels=self.class_names, yticklabels=self.class_names)
                # plt.xlabel("Predicted")
                # plt.ylabel("True")
                # writer.add_figure("Training Confusion Matrix", plt.gcf(), epoch)
                # plt.close()
                # print()
                # print("Validation:")
                    # val_iter.set_postfix(val_metrics["scalar"])

                
                
                # if SAVE_MODEL is not None:
                #     if (epoch + 1) % SAVE_MODEL == 0:
                #         curr_metrics = np.sum(list(scalar_metrics["val"].values()))
                #         if False:#curr_metrics < prev_metrics:
                #             print("Current Metrics not as good, skipping")
                #         else:
                #             prev_metrics = curr_metrics
                #             m_name = f"epoch_{epoch + 1}"
                #             for k in scalar_metrics["val"].keys():
                #                 m_name += f"_{k}_{scalar_metrics['val'][k]:.2f}"
                #             if not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}")):
                #                 os.makedirs(os.path.join(self.model_name, f"timesteps_{self.time_steps}"))
                #             if not os.path.exists(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}")):
                #                 os.makedirs(os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}"))
                #             filename = os.path.join(self.model_name, f"timesteps_{self.time_steps}", f"classes_{self.n_classes}", m_name+".pt")
                #             print(f"Saving model to {filename}")
                #             torch.save(self.classifier.state_dict(), filename)
                    
                # fig = plt.figure()
                # image = torch.image.decode_png(fig.getvalue(), channels=4)

                # sns.heatmap(val_metrics["image"]["val_conf_mat"], annot=False, xticklabels=self.class_names, yticklabels=self.class_names)
                # plt.xlabel("Predicted")
                # plt.ylabel("True")
                # plt.imshow(hm)
                # quit()
                # hm = fig
                # img_flat = np.frombuffer(fig.canvas.draw().tostring_rgb(), dtype='uint8')
                # image = img_flat.reshape(*reversed(img_flat.get_width_height), 3)
                # writer.add_figure("Validation Confusion Matrix", plt.gcf(), epoch)
                # plt.close()
                # print("##################")
                # for k in scalar_metrics["train"].keys():
                #     v = scalar_metrics["train"][k]
                #     v_val = scalar_metrics["val"]["val_" + k]
                    # print(f"{k.capitalize()}: {v:.3f}")
                    # print(f"{('val_' + k).capitalize()}: {v_val:.3f}")
                    # print("##################")
                    # writer.add_scalars(k.capitalize(), {"train": v,
                    #                                     "validation":v_val
                    # }, epoch)
                checkpoint_data = {
                    "epoch":epoch,
                    "net_state_dict": self.classifier.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict()
                    
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    data_path = Path(checkpoint_dir) / "data.pkl"
                    with open(data_path, "wb") as fp:
                        pickle.dump(checkpoint_data, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report(scalar_metrics["val"], checkpoint=checkpoint)

        if TUNE:
            config = {
                "l1": tune.choice([i for i in range(3)]),
                "l2": tune.choice([i for i in range(3)]),
                "l3": tune.choice([i for i in range(3)]),
                "lr": tune.loguniform(1e-5, 1e-1),
                "dropout": tune.loguniform(1e-1, 0.9)
            }


            tuner = Tuner(
                trainable=tune.with_parameters(tune.with_resources(tune_hyperparams, {"gpu": 1}), data=(self.total_train_set, self.total_val_set)),
                param_space=config,
                tune_config=tune.TuneConfig(
                    num_samples=30,
                    scheduler=tune.schedulers.ASHAScheduler(metric="val_loss", mode="min", time_attr='epoch', max_t=30)
                )
            )
            result = tuner.fit()

            # result = tune.run(
            #     partial(tune_hyperparams, data_dir="tuning"),
            #     resources_per_trial={"cpu": 1, "gpu": 1},
            #     config=config,
            #     scheduler=scheduler
            # )
            print(dir(result.get_best_result()))
            best_trial = result.get_best_result()
            print(f"Best trial config: \t {best_trial.config}")
            print(f"Best Trial Final Validation Metrics: \t {best_trial.metrics_dataframe}")

            best_trained_model = InceptionI3dGraph(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, l1=best_trial.config["l1"], l2=best_trial.config["l2"], l3=best_trial.config["l3"], dropout=best_trial.config["dropout"]).to(device)

            best_checkpoint = best_trial.get_best_checkpoint(trial=best_trial, metric="val_accuracy", mode="max")

            with best_checkpoint.as_directory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with open(data_path, "rb") as fp:
                    best_checkpoint_data = pickle.load(fp)

                best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        else:
          
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