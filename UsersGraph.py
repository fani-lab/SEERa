import networkx as nx
import matplotlib.pyplot as plt
import time
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def CreateUsersGraph(day, users, users_topic_interests):
    num_users = len(users_topic_interests)
    print('There are', num_users, ' users on', day)
    print('salam')
    if num_users < 1:
        return -1
    num_topics = users_topic_interests[0].shape[0]
    tic = time.process_time()
    usersSimilarity = np.round(cosine_similarity(users_topic_interests), 3)
    print('Similarity Matrix shape is', usersSimilarity.shape)
    toc = time.process_time()
    # usersSimilarity = np.zeros((num_users, num_users))
    # tic = time.process_time()
    # for u1idx in range(num_users):
    #     for u2idx in range(u1idx, num_users):
    #         sim = Similarity(users_topic_interests[u1idx], users_topic_interests[u2idx])
    #         usersSimilarity[u1idx, u2idx] = sim
    #         usersSimilarity[u2idx, u1idx] = sim
    # toc = time.process_time()
    print('Similarity calculation takes', toc - tic, 'second for', num_users, 'users.')
    print('Similarity Matrix is generated for', day)
    tic = time.process_time()
    G = nx.from_numpy_matrix(usersSimilarity)
    toc = time.process_time()
    print('Generating graph takes', toc - tic, 'second for', num_users, 'users.')

    # layout = nx.spring_layout(G)
    # nx.draw(G)
    # nx.draw_networkx_edge_labels(G, pos=layout)
    # plt.savefig('graph_'+str(day)+'_'+str(num_users)+'users_'+str(num_topics)+'topics.png')
    # plt.close()
    nx.write_pajek(G, 'graph_ '+str(num_users)+'users_'+str(num_topics)+'topics.net')
    return G


def Similarity(interests1, interests2, sim_type='cos'):
    if sim_type == 'cos':
        i1 = sparse.csr_matrix(interests1)
        i2 = sparse.csr_matrix(interests2)
        sim = np.round(cosine_similarity(i1, i2), 3)[0][0]
        if sim < 0.5:
            sim = 0
        return sim
    else:
        pass