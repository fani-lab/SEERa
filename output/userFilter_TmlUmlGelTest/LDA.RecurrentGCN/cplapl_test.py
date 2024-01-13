import pandas as pd
import numpy as np


puc = np.load('cpl/PredUserClusters.npy', allow_pickle=True)
uc = pd.read_csv('cpl/user_clusters.csv')

community_recommendations = pd.read_csv('apl/community_recommendations.csv')
user_community_recommendations = pd.read_csv('apl/user_community_recommendations.csv')