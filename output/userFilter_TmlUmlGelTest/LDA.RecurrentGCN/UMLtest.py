import numpy as np
import pandas as pd
import pickle
import os
import torch

doc = pd.read_csv('uml/documents.csv')
graphs = torch.load('uml/graphs/graphs.pt')
features = np.load('uml/user_interests/features.npy')

for i in range(len(graphs.features)):
    assert (graphs.features[i] == features[i]).all()

for i in range(len(graphs.targets)):
    assert (graphs.targets[i] == features[i + 1]).all()
