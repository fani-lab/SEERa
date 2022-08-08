import networkx as nx
from scipy import sparse
#import numpy as np
#import sys
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

import params
from cmn import Common as cmn

def create_users_graph(day, users_topic_interests, path_2_save):
    num_users = len(users_topic_interests)
    cmn.logger.info(f'UsersGraph: There are {num_users} users on {day}')
    if num_users < 1:
        return -1
    num_topics = users_topic_interests.shape[1]
    user_similarity_threshold = params.uml['userSimilarityThreshold']
    users_topic_interests = users_topic_interests.sort_index(axis=1)
    users_topic_interests = sparse.csr_matrix(users_topic_interests)
    # usersSimilarity = cosine_similarity(users_topic_interests_sparse)
    users_similarity = pairwise_kernels(users_topic_interests.T, metric='cosine', n_jobs=9)
    users_similarity[users_similarity < user_similarity_threshold] = 0
    users_similarity = sparse.csr_matrix(users_similarity)
    g = nx.from_scipy_sparse_matrix(users_similarity, parallel_edges=False, create_using=None, edge_attribute='weight')
    # nx.write_pajek(G, f'{path_2_save}/graph_{num_users}users_{num_topics}topics_{day}day.net')
    return g
