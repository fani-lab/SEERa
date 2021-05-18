import networkx as nx
from scipy import sparse
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

sys.path.extend(["../"])
from cmn import Common as cmn

def create_users_graph(day, users_topic_interests, path_2_save):
    num_users = len(users_topic_interests)
    cmn.logger.info(f'UsersGraph: There are {num_users} users on {day}')
    if num_users < 1:
        return -1
    num_topics = users_topic_interests.shape[1]
    usersSimilarity = cosine_similarity(users_topic_interests)
    G = nx.from_numpy_matrix(usersSimilarity)
    nx.write_pajek(G, f'{path_2_save}/graph_{num_users}users_{num_topics}topics_{day}day.net')
    return G


def similarity(interests1, interests2, sim_type='cos'):
    if sim_type == 'cos':
        i1 = sparse.csr_matrix(interests1)
        i2 = sparse.csr_matrix(interests2)
        sim = np.round(cosine_similarity(i1, i2), 3)[0][0]
        if sim < 0.1:
            sim = 0
        return sim
    else:
        pass