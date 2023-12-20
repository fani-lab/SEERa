import torch
import torch.nn as nn
from torch_geometric_temporal.nn import GCLSTM
from torch_geometric.data import Dataset, Data, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import GConvGRU
import params
class MyTemporalGraphDataset(Dataset):
    def __init__(self, train_graphs, target_graph):
        self.train_graphs = train_graphs
        self.target_graph = target_graph

    def __len__(self):
        return len(self.train_graphs)

    def __getitem__(self, idx):
        target_edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        data = self.train_graphs[idx]
        data.edge_attr = torch.zeros(data.edge_index.shape[1], 1)
        return data, target_edges

class MyTemporalGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyTemporalGraphModel, self).__init__()
        self.lstm = GCLSTM(input_dim, hidden_dim, K=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.lstm(x, edge_index)
        output = self.fc(h)
        return output


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, 2)
        self.linear = torch.nn.Linear(filters, node_features)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

def modelTrain(dataset):
    print('Asssal')
    model = RecurrentGCN(node_features=params.tml['numTopics'], filters=8)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    model.train()
    from tqdm import tqdm
    for epoch in tqdm(range(num_epochs)):
        for time, snapshot in enumerate(dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = torch.mean((y_hat-snapshot.y)**2)
            cost.backward()
            optimizer.step()
            optimizer.zero_grad()
    model.eval()
    cost = 0
    for time, snapshot in enumerate(dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        print(cost)
    cost = cost / (time+1)
    cost = cost.item()
    print("MSE: {:.4f}".format(cost))
    return model


def main(dataset):
    model = modelTrain(dataset)
    model.eval()
    with torch.no_grad():
        predicted_features = model(dataset[-1].y, dataset[-1].edge_index, dataset[-1].edge_weight)

    return predicted_features