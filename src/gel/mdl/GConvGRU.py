from torch_geometric_temporal.nn.recurrent import GConvGRU
import torch
import torch.nn.functional as F

class model(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(model, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, node_features)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h