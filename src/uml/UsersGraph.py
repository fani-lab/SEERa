import torch
import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as sgts

import params
from cmn import Common as cmn

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, num_timesteps, features, edge_index):
        self.num_nodes, self.num_features, self.num_timesteps = features.shape[0], features.shape[1], num_timesteps
        self.edge_index = torch.unique(edge_index[:, edge_index[0] != edge_index[1]], dim=1)
        self.edges_weight = torch.ones(self.edge_index.shape[1])
        x_list = [features[t] for t in range(num_timesteps - 1)]
        y_list = [features[t + 1] for t in range(num_timesteps - 1)]
        self.data = sgts(np.asarray(self.edge_index), np.asarray(self.edges_weight), x_list, y_list)
    def __len__(self):
        return self.num_nodes
    def __getitem__(self, idx):
        return self.data
def graph_generator(documents):
    all_nodes =  documents['UserId'].unique()
    num_random_edges = 5 * len(all_nodes)
    edges = torch.randint(0, len(all_nodes), (2, num_random_edges), dtype=torch.long)
    edges = edges[:, edges[0] != edges[1]]
    edges = torch.unique(edges, dim=1)
    num_timeStamps = documents['TimeStamp'].max() - documents['TimeStamp'].min() + 1
    features = np.zeros((num_timeStamps, len(all_nodes), params.tml['numTopics']))
    for t in range(num_timeStamps):
        documents_t = documents[documents['TimeStamp'] == t]
        user_ids_t = documents_t['UserId'].values
        indices = np.isin(all_nodes, user_ids_t)
        features[t, indices] = np.asarray([eval(documents_t[documents_t['UserId'] == user_id]['TopicInterests'].iloc[0]) for user_id in all_nodes[indices]])
    dataset = MyDataset(num_timeStamps, features, edges)
    import pickle
    with open(f"{params.uml['path2save']}/graphs/graphs.pkl", 'wb') as f:
        pickle.dump(dataset.data, f)
    torch.save(dataset.data, f"{params.uml['path2save']}/graphs/graphs.pt")
    with open(f"{params.uml['path2save']}/user_interests/features.pkl", 'wb') as f:
        pickle.dump(features, f)
    np.save(f"{params.uml['path2save']}/user_interests/features.npy",features)
    return dataset.data