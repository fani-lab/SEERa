import pickle, os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import GConvGRU
import params

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
    model = RecurrentGCN(node_features=params.tml['numTopics'], filters=8)
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

def main(documents, dataset):
    if not os.path.isdir(params.gel["path2save"]): os.makedirs(params.gel["path2save"])
    model = modelTrain(dataset)
    model.eval()
    with torch.no_grad():
        predicted_features = model(dataset[-1].y, dataset[-1].edge_index, dataset[-1].edge_weight)
    print(predicted_features)
    with open(f'{params.gel["path2save"]}/embeddings.pkl', 'wb') as f:
        pickle.dump(predicted_features, f)
    torch.save(predicted_features, f'{params.gel["path2save"]}/embeddings.pt')
    unique_users = documents['UserId'].unique()
    user_features = pd.DataFrame({'UserId': unique_users, 'FinalInterests': predicted_features.tolist()})
    user_features.to_csv(f'{params.gel["path2save"]}/userFeatures.csv')
    return user_features, predicted_features