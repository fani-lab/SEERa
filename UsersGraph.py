import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def CreateUsersGraph(day, users, users_topic_interests):
    num_users = len(users_topic_interests)
    if num_users < 1:
        return -1
    num_topics = users_topic_interests[0].shape[0]
    usersSimilarity = np.zeros((num_users, num_users))
    for u1idx in range(num_users):
        for u2idx in range(u1idx, num_users):
            sim = Similarity(users_topic_interests[u1idx], users_topic_interests[u2idx])
            usersSimilarity[u1idx,u2idx] = sim
            usersSimilarity[u2idx, u1idx] = sim
    G = nx.from_numpy_matrix(usersSimilarity)
    layout = nx.spring_layout(G)
    nx.draw(G)
    nx.draw_networkx_edge_labels(G, pos=layout)
    plt.savefig('graph_'+str(day)+'_'+str(num_users)+'users_'+str(num_topics)+'topics.png')
    nx.write_pajek(G, 'graph_ '+str(num_users)+'users_'+str(num_topics)+'topics.net')
    return G


def Similarity(interests1, interests2, sim_type='cos'):
    if sim_type == 'cos':
        return np.round(cosine_similarity([interests1], [interests2]), 3)
    else:
        pass