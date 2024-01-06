import networkx as nx
from scipy import sparse
#import numpy as np
#import sys
import pandas as pd
import torch
import numpy as np
import torch_geometric
import torch_geometric_temporal
from torch_geometric_temporal.signal import StaticGraphTemporalSignal as sgts
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

import params
from cmn import Common as cmn

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, num_timesteps, features, edge_index):
        self.num_nodes = features.shape[0]
        self.num_features = features.shape[1]
        self.num_timesteps = num_timesteps
        self.edge_index = edge_index

        edges = edge_index
        # Remove self-loops and duplicate edges
        edges = edges[:, edges[0] != edges[1]]
        edges = torch.unique(edges, dim=1)

        edges_weight = torch.ones(edges.shape[1])
        x_list, y_list = [], []
        for t in range(num_timesteps-1):

            x_list.append(features[t])
            y_list.append(features[t+1])
        self.data = sgts(np.asarray(edges), np.asarray(edges_weight), x_list, y_list)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, idx):
        return self.data

def graph_generator(documents, connections_path):
    all_nodes =  documents['UserId'].unique()

    num_random_edges = 5 * len(all_nodes)
    edges = torch.randint(0, all_nodes.shape[0], (2, num_random_edges), dtype=torch.long)
    # Remove self-loops and duplicate edges
    edges = edges[:, edges[0] != edges[1]]
    edges = torch.unique(edges, dim=1)
    edges_weight = torch.ones(edges.shape[1])
    # generating feature matrix
    num_timeStamps = documents['TimeStamp'].max() - documents['TimeStamp'].min()
    num_features = params.tml['numTopics']

    features = np.zeros((num_timeStamps, all_nodes.shape[0], num_features))
    for t in range(num_timeStamps):
        documents_t = documents[documents['TimeStamp'] == t]
        for index, user_id in enumerate(all_nodes):
            if user_id in documents_t['UserId'].values:
                features[t][index] = np.asarray(eval(documents_t.loc[documents_t['UserId'] == user_id, 'TopicInterests'].iloc[0]))

    print (features.shape)
    dataset = MyDataset(num_timeStamps, features, edges)
    import pickle
    with open(f"{params.uml['path2save']}/graphs/graphs.pkl", 'wb') as f:
        pickle.dump(dataset.data, f)
    torch.save(dataset.data, f"{params.uml['path2save']}/graphs/graphs.pt")
    with open(f"{params.uml['path2save']}/user_interests/features.pkl", 'wb') as f:
        pickle.dump(features, f)
    np.save(f"{params.uml['path2save']}/user_interests/features.npy",features)
    return dataset.data