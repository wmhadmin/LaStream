import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict, Optional


class GCNMemory(nn.Module):
    """
    GCN-based memory network for extracting streaming application embeddings.
    This network processes graph structures representing task dependencies in streaming applications.
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dims: List[int] = [128, 256, 128],
                 output_dim: int = 64,
                 dropout: float = 0.1):
        super(GCNMemory, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.gcn_layers.append(GCNConv(dims[i], dims[i + 1]))
        
        # Final projection layer to compress to output dimension
        self.projection = nn.Linear(hidden_dims[-1], output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GCN network.
        
        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges] 
                - batch: Batch assignment for nodes (if batched)
        
        Returns:
            torch.Tensor: Graph embedding vector [output_dim]
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        
        # Apply GCN layers with ReLU activation and dropout
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            x = F.relu(x)
            if i < len(self.gcn_layers) - 1:  # Don't apply dropout to last layer
                x = self.dropout_layer(x)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final projection
        x = self.projection(x)
        
        return x.squeeze(0) if batch is None else x


def create_wordcount_graph(num_operators: int = 4) -> Data:
    """
    Create a WordCount streaming application graph as an example.
    
    WordCount topology:
    - Source (Spout) -> Splitter -> Counter -> Sink
    
    Args:
        num_operators: Number of operators in the WordCount application
        
    Returns:
        Data: PyTorch Geometric Data object representing the WordCount graph
    """
    # WordCount operators: [Source, Splitter, Counter, Sink]
    operator_names = ["Source", "Splitter", "Counter", "Sink"]
    
    # Node features: [processing_capability, input_rate, resource_load, selectivity]
    node_features = torch.tensor([
        [1.0, 0.8, 0.3, 1.0],  # Source: high processing, high input rate
        [0.9, 0.8, 0.4, 5.0],  # Splitter: splits words, high selectivity
        [0.7, 4.0, 0.6, 0.2],  # Counter: groups/counts, low selectivity
        [1.0, 0.8, 0.2, 1.0],  # Sink: output, stable load
    ], dtype=torch.float32)
    
    # Pad features to input_dim if necessary
    if node_features.size(1) < 64:
        padding = torch.zeros(num_operators, 64 - node_features.size(1))
        node_features = torch.cat([node_features, padding], dim=1)
    
    # Edge connectivity: Source->Splitter->Counter->Sink
    edge_index = torch.tensor([
        [0, 1, 2],  # Source nodes
        [1, 2, 3]   # Target nodes
    ], dtype=torch.long)
    
    return Data(x=node_features[:num_operators], edge_index=edge_index)


def create_adjacency_matrix(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """
    Create adjacency matrix from edge index.
    
    Args:
        edge_index: Edge connectivity tensor [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        np.ndarray: Adjacency matrix [num_nodes, num_nodes]
    """
    adj_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj_matrix[src][dst] = 1
        
    return adj_matrix


class StreamingApplicationEncoder:
    """
    Helper class to encode streaming applications for GCN processing.
    """
    
    def __init__(self, gcn_memory: GCNMemory):
        self.gcn_memory = gcn_memory
    
    def encode_application(self, 
                          operators_info: List[Dict],
                          dependencies: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Encode a streaming application into an embedding vector.
        
        Args:
            operators_info: List of operator information dictionaries containing:
                - processing_capability: Average processing capability
                - input_rate: Total input rate
                - resource_load: Resource utilization
                - dependencies: Dependency relationships
            dependencies: List of (source_id, target_id) tuples representing dependencies
            
        Returns:
            torch.Tensor: Application embedding vector
        """
        num_operators = len(operators_info)
        
        # Create node features matrix
        features = []
        for op_info in operators_info:
            feature_vector = [
                op_info.input_rate,
                op_info.processing_capacity,
                op_info.cpu_usage,
                op_info.memory_usage
            ]
            # Pad to required dimension
            while len(feature_vector) < self.gcn_memory.input_dim:
                feature_vector.append(0.0)
            features.append(feature_vector[:self.gcn_memory.input_dim])
        
        node_features = torch.tensor(features, dtype=torch.float32)
        
        # Create edge index
        if dependencies:
            edge_index = torch.tensor([
                [dep[0] for dep in dependencies],
                [dep[1] for dep in dependencies]
            ], dtype=torch.long)
        else:
            # Create empty edge index if no dependencies
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index)
        
        # Extract embedding
        self.gcn_memory.eval()
        with torch.no_grad():
            embedding = self.gcn_memory(data)
        
        return embedding