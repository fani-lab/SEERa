import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

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


def communities_topic_interest(user_clusters, news_topics):
    num_topics = news_topics.shape[1]
    users_topic_interests_list = sorted(glob.glob(f'{Params.uml["path2save"]}/Day*UsersTopicInterests.npy'))

    last_UTI = np.load(users_topic_interests_list[-1]).T
    communities_topic_interests = []
    cluster_numbers = []
    for uc in range(user_clusters.min(), user_clusters.max()+1):
        users_in_cluster = np.where(user_clusters == uc)[0]
        if len(users_in_cluster) < Params.cpl["minSize"]: break
        topic_interest_sum = np.zeros(num_topics)
        for user in users_in_cluster: topic_interest_sum += last_UTI[user]
        communities_topic_interests.append(topic_interest_sum)
        cluster_numbers.append(uc)
    communities_topic_interests = np.asarray(communities_topic_interests)
    np.save(f'{Params.apl["path2save"]}/CommunitiesTopicInterests.npy', communities_topic_interests)
    np.save(f'{Params.apl["path2save"]}/ClusterNumbers.npy', cluster_numbers)
    news = np.zeros((len(news_topics), num_topics))
    for nt in range(len(news_topics)):
        news_vector = np.asarray(news_topics[nt])
        news_vector_temp = np.zeros(num_topics)
        counter = 0
        for topic in news_vector:
            news_vector_temp[counter] = topic
            counter += 1
        news[nt] = news_vector_temp
    return communities_topic_interests


def recommend(communities_topic_interests, news, news_ids, topK):
    recommendation_table = np.matmul(communities_topic_interests, news.T)
    top_recommendations = np.zeros((len(communities_topic_interests), topK))
    for r in range(len(recommendation_table)):
        news_scores = recommendation_table[r]
        sorted_index_array = np.argsort(news_scores)
        sorted_array = news_ids[sorted_index_array]
        top_recommendations[r] = np.flip(sorted_array[-topK:])


    # recommendation_table_analyzer(recommendation_table, 'NRN', 'CommunityPerNewsNumbers')
    # recommendation_table_analyzer(recommendation_table, 'CRN', 'NewsPerCommunityNumbers')
    np.save(f'{Params.apl["path2save"]}/TopRecommendationsCluster.npy', top_recommendations)
    np.save(f'{Params.apl["path2save"]}/RecommendationTableCluster.npy', recommendation_table)
    return top_recommendations

def user_recommend(pred_user_clusters, top_recommendations):
    users = np.load(f'{Params.uml["path2save"]}/users.npy')
    user_recommendation = {}
    for u in range(len(users)):
        cluster = pred_user_clusters[u]
        try:
            user_recommendation[users[u]] = list(top_recommendations[cluster])
        except:
            continue
    f = open(f'{Params.apl["path2save"]}/TopRecommendationsUser.pkl', "wb")
    pickle.dump(user_recommendation, f)
    f.close()
    return user_recommendation

def internal_test(top_recommendations):
    print('Top Recommendation test:')
    a = top_recommendations.reshape(-1)
    b = set(a)
    print(len(a))
    print(len(b))
    print(len(a)/len(b))
    duplicate = False
    for i in top_recommendations:
        if len(i) != len(set(i)):
            duplicate = True
            print(i)
    if not duplicate:
        print('All rows has distinct news Ids.', len(i))
    print('Top Recommendation shape: ', top_recommendations.shape)


def main(news_topics, top_k=10):
    news = pd.read_csv(f'{Params.dal["path"]}/News.csv')
    news_ids = news["NewsId"]
    user_clusters = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
    communities_topic_interests = communities_topic_interest(user_clusters, news_topics)
    top_recommendations = recommend(communities_topic_interests, news_topics, news_ids, top_k)
    return user_recommend(user_clusters, top_recommendations)


