import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params
from cmn import Common as cmn

def recommendation_table_analyzer(RT, test, savename):
    if test == 'CRN': pass# 'CRN' = 'community recommendations number'
    elif test == 'NRN': RT = RT.T# 'news recommendations number'
    comm_news = []
    for com in RT: comm_news.append(com.sum())
    if test == 'NRN': comm_news.sort()
    plt.plot(range(len(comm_news)), comm_news)
    plt.savefig(f'{params.apl["path2save"]}/{savename}.png')
    plt.close()
    return comm_news

def communities_topic_interest(UserClusters, NewsTopics):
    num_topics = NewsTopics.shape[1]
    UsersTopicInterestsList = sorted(glob.glob(f'{params.uml["path2save"]}/Day*UsersTopicInterests.npy'))
    LastUTI = np.load(UsersTopicInterestsList[-1])
    CommunitiesTopicInterests = []
    ClusterNumbers = []
    for UC in range(UserClusters.min(), UserClusters.max()+1):
        UsersinCluster = np.where(UserClusters == UC)[0]
        if len(UsersinCluster) < params.cpl["minSize"]: break
        TopicInterestSum = np.zeros(num_topics)
        for user in UsersinCluster: TopicInterestSum += LastUTI[user]
        CommunitiesTopicInterests.append(TopicInterestSum)
        ClusterNumbers.append(UC)

    CommunitiesTopicInterests = np.asarray(CommunitiesTopicInterests)
    np.save(f'{params.apl["path2save"]}/CommunitiesTopicInterests.npy', CommunitiesTopicInterests)
    #cmn.save2excel(CommunitiesTopicInterests, 'evl/CommunitiesTopicInterests')
    np.save(f'{params.apl["path2save"]}/ClusterNumbers.npy', ClusterNumbers)
    #cmn.save2excel(ClusterNumbers, 'evl/ClusterNumbers')
    News = np.zeros((len(NewsTopics), num_topics))
    for NT in range(len(NewsTopics)):
        NewsVector = np.asarray(NewsTopics[NT])
        NewsVector_temp = np.zeros(num_topics)
        counter = 0
        for topic in NewsVector:
            NewsVector_temp[counter] = topic
            counter += 1
        News[NT] = NewsVector_temp
    # np.save(f'{params.apl["path2save"]}/NewsTopicInterests.npy', News)
    return CommunitiesTopicInterests

def recommend(communities_topic_interests, news, news_ids, topK):
    recommendation_table = np.matmul(communities_topic_interests, news.T)
    #cmn.save2excel(RecommendationTable, 'evl/RecommendationTable')
    top_recommendations = np.zeros((len(communities_topic_interests), topK))
    for r in range(len(recommendation_table)):
        news_scores = recommendation_table[r]
        top_score = np.partition(news_scores.flatten(), -topK)[-topK]
        recommendation_candidates = np.where(news_scores >= top_score)[0]
        recommendation_scores = news_scores[recommendation_candidates]
        inds = np.flip(recommendation_scores.argsort())
        # Recommendations_sorted = RecommendationCandidates[inds]
        recommendations_sorted = news_ids[inds]
        top_recommendations[r] = recommendations_sorted[:topK]

    recommendation_table_analyzer(recommendation_table, 'NRN', 'CommunityPerNewsNumbers')
    recommendation_table_analyzer(recommendation_table, 'CRN', 'NewsPerCommunityNumbers')
    np.save(f'{params.apl["path2save"]}/TopRecommendationsCluster.npy', top_recommendations)
    np.save(f'{params.apl["path2save"]}/RecommendationTableCluster.npy', recommendation_table)
    return top_recommendations

def user_recommend(pred_user_clusters, top_recommendations):
    #pred_user_clusters = np.load('../cpl/PredUserClusters.npy')
    users = np.load(f'{params.uml["path2save"]}/users.npy')
    user_recommendation = {}
    for u in range(len(users)):
        cluster = pred_user_clusters[u]
        try:
            user_recommendation[users[u]] = list(top_recommendations[cluster])
        except:
            continue
    np.save(f'{params.apl["path2save"]}/TopRecommendationsUser.npy', pd.DataFrame(user_recommendation))
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
    news = pd.read_csv(f'{params.dal["toyPath"]}/News.csv')
    news_ids = news["NewsId"]
    user_clusters = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    communities_topic_interests = communities_topic_interest(user_clusters, news_topics)
    top_recommendations = recommend(communities_topic_interests, news_topics, news_ids, top_k)
    return user_recommend(user_clusters,top_recommendations)


