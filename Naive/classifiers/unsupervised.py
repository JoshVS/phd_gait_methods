import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        # x shape: (Batch, Seq_Len, 34)
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