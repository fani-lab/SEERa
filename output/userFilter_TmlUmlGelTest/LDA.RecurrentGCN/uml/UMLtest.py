import numpy as np
import pandas as pd
import pickle
import os
import torch

doc = pd.read_csv('documents.csv')
graphs = torch.load('graphs/graphs.pt')
features = np.load('user_interests/features.npy')

for i in range(len(graphs.features)):
    assert (graphs.features[i] == features[i]).all()

for i in range(len(graphs.targets)):
    assert (graphs.targets[i] == features[i + 1]).all()