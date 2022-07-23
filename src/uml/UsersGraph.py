import networkx as nx
from scipy import sparse
#import numpy as np
#import sys
import params
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels

from cmn import Common as cmn

def create_users_graph(day, users_topic_interests, path_2_save):
    num_users = len(users_topic_interests)
    cmn.logger.info(f'UsersGraph: There are {num_users} users on {day}')
    if num_users < 1:
        return -1
    num_topics = users_topic_interests.shape[1]
    UserSimilarityThreshold = params.uml['UserSimilarityThreshold']
    users_topic_interests = sparse.csr_matrix(users_topic_interests)
    # usersSimilarity = cosine_similarity(users_topic_interests_sparse)
    usersSimilarity = pairwise_kernels(users_topic_interests, metric='cosine', n_jobs=9)
    usersSimilarity[usersSimilarity < UserSimilarityThreshold] = 0
    usersSimilarity = sparse.csr_matrix(usersSimilarity)
    G = nx.from_scipy_sparse_matrix(usersSimilarity, parallel_edges=False, create_using=None, edge_attribute='weight')
    # nx.write_pajek(G, f'{path_2_save}/graph_{num_users}users_{num_topics}topics_{day}day.net')
    return G
