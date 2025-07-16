import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=(9, 1)):
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
        res = self.residual(x.view(N * V, C, T))
        
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