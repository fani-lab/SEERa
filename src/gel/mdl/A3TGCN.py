import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class model(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(model, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

# model(node_features=2, periods=12)