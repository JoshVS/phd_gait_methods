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

class MemoryModule(nn.Module):
    """
    Memory Module based on MemAE (Gong et al. 2019).
    Stores prototypical patterns of normal behavior.
    """
    def __init__(self, mem_dim, feat_dim, shrink_thres=0.0025):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim  # Number of memory slots (N)
        self.feat_dim = feat_dim  # Dimension of each slot (C)
        self.shrink_thres = shrink_thres
        
        # Trainable memory matrix [2]
        self.memory = nn.Parameter(torch.Tensor(mem_dim, feat_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memory.size(1))
        self.memory.data.uniform_(-stdv, stdv)

    def hard_shrinkage(self, w):
        """Induces sparsity in memory addressing weights."""
        w = F.relu(w - self.shrink_thres)
        return w / (torch.abs(w).sum(dim=-1, keepdim=True) + 1e-12)

    def forward(self, z):
        # Calculate similarity between query z and memory items
        # z shape: (Batch, Seq, Feat_Dim)
        # memory shape: (Mem_Dim, Feat_Dim)
        attn_weights = F.linear(z, self.memory) # (B, S, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Apply sparsity-inducing shrinkage
        attn_weights = self.hard_shrinkage(attn_weights)
        
        # Retrieve patterns: z_hat = w * M [2]
        z_hat = torch.matmul(attn_weights, self.memory) # (B, S, C)
        
        return z_hat, attn_weights

class PoseTransformerMemAE(nn.Module):
    """
    Transformer-based Pose Anomaly Detector with Memory Augmentation.
    Optimized for skeletal sequences (17 keypoints = 34 dimensions).
    """
    def __init__(self, num_joints=17, d_model=128, nhead=8, num_layers=4, mem_slots=100):
        super(PoseTransformerMemAE, self).__init__()
        self.input_dim = num_joints * 2  # (x, y) coordinates
        
        # Embedding and Positional Encoding
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model)) # Support up to 1000 frames
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Memory-Augmented Bottleneck [3]
        self.memory_module = MemoryModule(mem_slots, d_model)
        
        # Decoder for Pose Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.input_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len*num_people, 34)
        batch_size, seq_len, _ = x.size()
        
        # 1. Temporal Encoding
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        z = self.transformer_encoder(x)
        
        # 2. Memory Retrieval (Restricts model to 'normal' prototypes)
        z_hat, weights = self.memory_module(z)
        
        # 3. Reconstruction
        reconstructed_pose = self.decoder(z_hat) 
        
        return reconstructed_pose, weights

def loss_function(recon_x, x, weights):
    """
    Combined loss: Reconstruction Error + Entropy Loss (to encourage sparsity).
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x)
    
    # Entropy loss on weights to force concentrated attention [3]
    entropy_loss = -torch.mean(torch.sum(weights * torch.log(weights + 1e-12), dim=-1))
    
    return recon_loss + 0.0002 * entropy_loss 

class SuspiciousActivityMonitor:
    def __init__(self, ds=None, loss_fn=loss_function, model_name=MODEL_NAME, weights_path=None, timesteps=None, testing=False):
        
        torch.set_default_dtype(torch.float32)
        self.model_name = model_name

        if weights_path is not None and testing:
            self.graph = create_ultralytics_graph()
            model_weights = torch.load(weights_path, map_location=device )
            self.classifier = PoseTransformerMemAE().to(device)
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
            self.classifier = PoseTransformerMemAE().to(device)
            self.classifier.load_state_dict(model_weights)
            print(f"Loaded model from {weights_path}")
            return

        self.train_set, self.val_set = ds
        # print(len(self.train_set)//866)
        # quit()
        ds = self.train_set
        self.time_steps = ds.num_timesteps
        self.n_point = ds.n_point
        self.num_person = ds.num_person
        self.in_channels = ds.in_channels
        self.loss_fn = loss_fn

        self.graph = MediapipeGraph(self.n_point, ds.in_edge)

        # self.classifier = PoseTransformerMemAE(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph).to(device)
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
        X = sample
        X = X.to(device)
        # print(y.size())
        optimizer.zero_grad()
        recon, weights = classifier(X)
        # predictions = predictions.view(-1, predictions.size(2))  # Flatten the predictions to match the output shape of the classifier
        loss = self.loss_fn(recon, X, weights)

        loss.backward()

        optimizer.step()
        metrics["scalar"]['loss'] = loss.item()
                    
        return metrics

    def _val_step(self, sample, val_metrics, classifier=None):
        if classifier is None:
            classifier = self.classifier
        val_metrics["scalar"] = {}
        X = sample
        X = X.to(device)
        
        
        with torch.no_grad():
            recon, weights = classifier(X)
            # predictions = predictions.view(-1, predictions.size(2))  # Flatten the predictions to match the output shape of the classifier
            # print(y.dtype)
            # quit()
            val_loss = self.loss_fn(recon, X, weights)
            val_metrics["scalar"]["val_loss"] = val_loss.item()
        
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

        # print((t,h,w))
        # quit()

        # print((self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph, self.time_steps))
        # quit()

        self.classifier = PoseTransformerMemAE().to(device)        
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

            best_trained_model = PoseTransformerMemAE().to(device)

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
                # sum = 0
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         sum += train_hm[i][j]
                
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         train_hm[i][j] /= sum

                # sum = 0
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         sum += train_hm[i][j]
                
                # for i in range(len(train_hm)):
                #     for j in range(len(train_hm[i])):
                #         print(f"{train_hm[i][j]} / {sum} = {train_hm[i][j] / sum}")
                #         train_hm[i][j] /= sum

                

                if SAVE_MODEL is not None:
                    if epoch % SAVE_MODEL == 0:
                        torch.save(self.classifier.state_dict(), os.path.join("weights", "train", self.model_name, f"valloss_{scalar_metrics['val']['val_loss']}_timesteps_{self.time_steps}_model_epoch_{epoch}.pth"))
            torch.save(self.classifier.state_dict(), os.path.join("weights", "finished", WEIGHTS_PATH))