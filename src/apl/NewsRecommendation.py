import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Params
from cmn import Common as cmn

def recommendation_table_analyzer(rt, test, savename):
    if test == 'CRN': pass# 'CRN' = 'community recommendations number'
    elif test == 'NRN': rt = rt.T# 'news recommendations number'
    comm_news = []
    for com in rt: comm_news.append(com.sum())
    if test == 'NRN': comm_news.sort()
    plt.plot(range(len(comm_news)), comm_news)
    plt.savefig(f'{Params.apl["path2save"]}/{savename}.png')
    plt.close()
    return comm_news

def recommend(communities_topic_interests, news_topics, topK):
    topK = min(topK, len(news_topics))
    news_ids = np.asarray(list(news_topics.keys()))
    news_topics = pd.DataFrame(news_topics)
    communities_topic_interests.reset_index(drop=True, inplace=True)
    news_topics.reset_index(drop=True, inplace=True)
    recommendation_table = communities_topic_interests.dot(news_topics)
    top_recommendations = {}#np.zeros((len(communities_topic_interests), topK))
    for index, row in recommendation_table.iterrows():
        news_scores = row.values
        sorted_index_array = np.argsort(news_scores)
        sorted_array = news_ids[sorted_index_array]
        top_recommendations[row.index[index]] = np.flip(sorted_array[-topK:])
    pd.to_pickle(top_recommendations, f'{Params.apl["path2save"]}/TopRecommendationsCluster.pkl')
    pd.to_pickle(recommendation_table, f'{Params.apl["path2save"]}/RecommendationTableCluster.pkl')
    return top_recommendations

def user_recommend(pred_user_clusters, top_recommendations):
    users = np.load(f'{Params.uml["path2save"]}/Users.npy')
    user_recommendation = {}
    for u in range(len(users)):
        cluster = pred_user_clusters[u]
        try:
            user_recommendation[users[u]] = list(top_recommendations[cluster])
        except:
            continue
    pd.to_pickle(user_recommendation,f'{Params.apl["path2save"]}/TopRecommendationsUser.pkl')
    return user_recommendation

def main(news_topics, top_k=10):
    user_clusters = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
    communities_topic_interests = pd.read_pickle(f'{Params.cpl["path2save"]}/ClusterTopic.pkl')
    try:
        top_recommendations = pd.read_pickle(f'{Params.apl["path2save"]}/TopRecommendationsCluster.pkl')
    except:
        top_recommendations = recommend(communities_topic_interests, news_topics, top_k)
    try:
        return pd.read_pickle(f'{Params.apl["path2save"]}/TopRecommendationsUser.pkl')
    except:
        return user_recommend(user_clusters, top_recommendations)


