import networkx as nx
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def CreateUsersGraph(day, users_topic_interests):
    num_users = len(users_topic_interests)
    print('There are', num_users, ' users on', day)
    if num_users < 1:
        return -1
    num_topics = users_topic_interests.shape[1]
    usersSimilarity = cosine_similarity(users_topic_interests)
    G = nx.from_numpy_matrix(usersSimilarity)
    nx.write_pajek(G, 'graph_ '+str(num_users)+'users_'+str(num_topics)+'topics.net')
    return G


def Similarity(interests1, interests2, sim_type='cos'):
    if sim_type == 'cos':
        i1 = sparse.csr_matrix(interests1)
        i2 = sparse.csr_matrix(interests2)
        sim = np.round(cosine_similarity(i1, i2), 3)[0][0]
        if sim < 0.1:
            sim = 0
        return sim
    else:
        pass