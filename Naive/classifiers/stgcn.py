"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
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

SAVE_MODEL = 1
LOAD_MODEL = True
MODEL_NAME = "model_checkpoints"
TUNE = True
DROPOUT = 0.25
WEIGHT_DECAY = 1e-5

BATCH_SIZE=8

EPOCHS = 2
LR = 1e-5

def force_cudnn_initialization():
    if device == "cuda":
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()
# Hi Josh, 

# Please find attached model code for the stgcn model as well as the graph creation class for mediapipe. 
# The input shape for the st-gcn model is [batch_size, channels, number_of_frames, nodes, M]
# Channels is usually 3 or 2 depending if you dealing with 3d or 2d coordinates
# M is the number of people in the clip, for me it is always 1 person pose data I'm working with.

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


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_, dropout=DROPOUT):
        super(GraphConvolution, self).__init__()
        self.cuda_ = cuda_
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32))) #graph_attn is the neighbourhoods - how is it represented?
        nn.init.constant_(self.graph_attn, 1)
        self.A = Variable(torch.from_numpy(A.astype(np.float64)), requires_grad=False)

        # Create Convolutions for each neighbourhood
        self.num_subset = 3 # number of neighbourhoods
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1)) # different convolutions for each neighbourhood
            weights_init(self.g_conv[i], bs=self.num_subset)

        # Residual connections
        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        # Create batch norm layers and dropout
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch * people, channels, times, nodes)
        """
        N, C, T, V = x.size() # (batch, channels, timesteps, nodes)
        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A * self.graph_attn # apply neighbourhoods to edges
        hidden_ = None

        # Convolution for each neighbourhood
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V) #(batch, time * channel, nodes)

            # Find nodes for this neighbourhood
            # x_a: (batch, time * channel, nodes)
            # A[i]: (nodes)

            # output: (batch, channel, time, nodes)
            # Effect: applies normalisation
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ = self.dropout(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.t_conv, bs=1)
        weights_init(self.bn, bs=1)

    def forward(self, x):
        """
        X: Shape (batch, channel, time, nodes)
        """
        x = self.bn(self.t_conv(x))
        return x


class ST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_=False, stride=1, residual=True, dropout=DROPOUT):
        super(ST_GCN_block, self).__init__()

        self.gcn = GraphConvolution(in_channels, out_channels, A, cuda_, dropout=dropout)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        x: (batch * people, channels, times, nodes)
        """
        # Graph convolution -> time convolution
        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)


class MarcSTGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, graph, cuda_=torch.cuda.is_available(), l1=1, l2=1, l3=1, dropout=DROPOUT):
        super(MarcSTGCN, self).__init__()

        self.graph = graph

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        weights_init(self.data_bn, bs=1)
        layers = [ST_GCN_block(in_channels, 64, A, cuda_, residual=False, dropout=dropout)]
        layers += [ST_GCN_block(64, 64, A, cuda_, dropout=dropout)] * (l1 - 1)
        layers += [ST_GCN_block(64, 128, A, cuda_, stride=2, dropout=dropout)]
        layers += [ST_GCN_block(128, 128, A, cuda_, dropout=dropout)] * (l2 - 1)
        layers += [ST_GCN_block(128, 256, A, cuda_, stride=2, dropout=dropout)]
        layers += [ST_GCN_block(256, 256, A, cuda_, dropout=dropout)] * (l3 - 1)
        
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
            x = self.layers['layer' + str(i+1)](x)
        # N*M,C,T,V

        c_new = x.size(1) # infer new channel size
        x = x.view(N, M, c_new, -1) # (batch, people, new_channel_size, times * nodes)
        x = x.mean(3).mean(1) # Take mean across times*nodes and people
        # return softmax(self.fc(x), dim=1) # in shape: (batch, new_channel_size)
        return self.fc(x)

class STGCN:
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
        self.num_person = ds.num_person
        self.in_channels = ds.in_channels
        self.loss_fn = loss_fn

        self.graph = MediapipeGraph(self.n_point, ds.in_edge)

        # self.classifier = MarcSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph).to(device)
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
            "precision": lambda x, y: precision_score(x.cpu().numpy(), y.cpu().numpy(), average="macro", zero_division=0.0),
            "recall": lambda x, y: recall_score(x.cpu().numpy(), y.cpu().numpy(), average="macro", zero_division=0.0)
        }
        self.train()
        # print(train_data.dtype)
        # quit()

        

    def _train_step(self, sample, optimizer, metrics, val_sample=None, classifier=None):
        if classifier is None:
            classifier = self.classifier
        metrics["scalar"] = {}
        if val_sample is not None:
            val_metrics = {}
        val_metrics = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device)
        # print(y.size())
        optimizer.zero_grad()
        outputs = classifier(X)
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

    def _val_step(self, sample, val_metrics, classifier=None):
        if classifier is None:
            classifier = self.classifier
        val_metrics["scalar"] = {}
        X, y = sample
        X = X.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            val_out = classifier(X)
            predictions = torch.nn.functional.one_hot(val_out.argmax(axis=1), num_classes=self.n_classes)
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

        self.classifier = MarcSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, dropout=DROPOUT).to(device)        
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
                print(f"Epoch #{epoch + 1}: ")
                
                train_metrics = {"scalar": {}, "image": {}}
                val_metrics = {"scalar": {}, "image": {}}
                
                for idx, curr_sample in enumerate(train_set):
                    train_metrics = self._train_step(ray.get(curr_sample), optimizer, train_metrics, classifier=classifier)
                for idx, val_sample in enumerate(val_set):
                    val_metrics = self._val_step(ray.get(val_sample), val_metrics, classifier=classifier)
        
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

            best_trained_model = MarcSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, dropout=best_trial.config["dropout"]).to(device)

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
                sns.heatmap(train_metrics["image"]["conf_mat"], annot=False, xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                writer.add_figure("Training Confusion Matrix", plt.gcf(), epoch)
                plt.close()

                
                sns.heatmap(val_metrics["image"]["val_conf_mat"], annot=False, xticklabels=self.class_names, yticklabels=self.class_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                writer.add_figure("Validation Confusion Matrix", plt.gcf(), epoch)