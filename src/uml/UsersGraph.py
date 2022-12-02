from scipy import sparse
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import binarize
import numpy as np

import Params
from cmn import Common as cmn

def create_users_graph(day, users_topic_interests, path_2_save):
    num_users = users_topic_interests.shape[1]
    cmn.logger.info(f'UsersGraph: There are {num_users} users on {day.date()}')
    if num_users < 1: return -1
    uti_sparse = sparse.csr_matrix(users_topic_interests)
    users_similarity = pairwise_kernels(uti_sparse.T, metric='cosine', dense_output=False)
    # users_similarity = pairwise_kernels(users_topic_interests.astype(np.float16).T, metric='cosine', n_jobs=9)
    #users_similarity = pairwise_kernels(users_topic_interests.T, metric='cosine', n_jobs=9)
    binarize(users_similarity, threshold=Params.uml['userSimilarityThreshold'], copy=False)
    sparse.save_npz(f'{path_2_save}/graphs/Day{day.date()}userSimilarities.npz', users_similarity)
