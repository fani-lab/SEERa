from scipy import sparse
from sklearn.metrics.pairwise import pairwise_kernels

import Params
from cmn import Common as cmn

def create_users_graph(day, users_topic_interests, path_2_save):
    num_users = len(users_topic_interests.T)
    cmn.logger.info(f'UsersGraph: There are {num_users} users on {day.date()}')
    if num_users < 1:
        return -1
    users_similarity = pairwise_kernels(users_topic_interests.T, metric='cosine', n_jobs=9)
    users_similarity[users_similarity < Params.uml['userSimilarityThreshold']] = 0
    users_similarity = sparse.csr_matrix(users_similarity)
    sparse.save_npz(f'{path_2_save}/graphs/Day{day.date()}userSimilarities.npz', users_similarity)
