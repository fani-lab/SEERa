import glob
import numpy as np
import matplotlib.pyplot as plt

import params
from cmn import Common as cmn

<<<<<<< HEAD
def recommendation_table_analyzer(RT, test, savename):
=======
def RecommendationTableAnalyzer(RT, test, savename):
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
    if test == 'CRN': pass# 'CRN' = 'community recommendations number'
    elif test == 'NRN': RT = RT.T# 'news recommendations number'
    comm_news = []
    for com in RT: comm_news.append(com.sum())
    if test == 'NRN': comm_news.sort()
    plt.plot(range(len(comm_news)), comm_news)
    plt.savefig(f'{params.apl["path2save"]}/{savename}.png')
    plt.close()
    return comm_news

<<<<<<< HEAD
def communities_topic_interest(UserClusters, NewsTopics):
=======
def CommunitiesTopicInterest(UserClusters, NewsTopics):
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
    num_topics = NewsTopics.shape[1]
    UsersTopicInterestsList = sorted(glob.glob(f'{params.uml["path2save"]}/Day*UsersTopicInterests.npy'))
    LastUTI = np.load(UsersTopicInterestsList[-1])
    CommunitiesTopicInterests = []
    ClusterNumbers = []
    for UC in range(UserClusters.min(), UserClusters.max()+1):
        UsersinCluster = np.where(UserClusters == UC)[0]
<<<<<<< HEAD
        if len(UsersinCluster) < params.cpl["minSize"]: break
=======
        if len(UsersinCluster) < params.cpl["min_size"]: break
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
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

<<<<<<< HEAD
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
    np.save(f'{params.apl["path2save"]}/TopRecommendations.npy', top_recommendations)
    np.save(f'{params.apl["path2save"]}/RecommendationTable.npy', recommendation_table)
    return top_recommendations


def internal_test(top_recommendations):
    print('Top Recommendation test:')
    a = top_recommendations.reshape(-1)
=======
def Recommend(CommunitiesTopicInterests, News, NewsIds, topK):
    RecommendationTable = np.matmul(CommunitiesTopicInterests, News.T)
    #cmn.save2excel(RecommendationTable, 'evl/RecommendationTable')
    TopRecommendations = np.zeros((len(CommunitiesTopicInterests), topK))
    for r in range(len(RecommendationTable)):
        NewsScores = RecommendationTable[r]
        topScore = np.partition(NewsScores.flatten(), -topK)[-topK]
        RecommendationCandidates = np.where(NewsScores >= topScore)[0]
        RecommendationScores = NewsScores[RecommendationCandidates]
        inds = np.flip(RecommendationScores.argsort())
        # Recommendations_sorted = RecommendationCandidates[inds]
        Recommendations_sorted = NewsIds[inds]
        TopRecommendations[r] = Recommendations_sorted[:topK]

    RecommendationTableAnalyzer(RecommendationTable, 'NRN', 'CommunityPerNewsNumbers')
    RecommendationTableAnalyzer(RecommendationTable, 'CRN', 'NewsPerCommunityNumbers')
    np.save(f'{params.apl["path2save"]}/TopRecommendations.npy', TopRecommendations)
    np.save(f'{params.apl["path2save"]}/RecommendationTable.npy', RecommendationTable)
    return TopRecommendations


def InternalTest(TopRecommendations):
    print('Top Recommendation test:')
    a = TopRecommendations.reshape(-1)
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
    b = set(a)
    print(len(a))
    print(len(b))
    print(len(a)/len(b))
    duplicate = False
<<<<<<< HEAD
    for i in top_recommendations:
=======
    for i in TopRecommendations:
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
        if len(i) != len(set(i)):
            duplicate = True
            print(i)
    if not duplicate:
        print('All rows has distinct news Ids.', len(i))
<<<<<<< HEAD
    print('Top Recommendation shape: ', top_recommendations.shape)


def main(news_topics, topK=10):
    news_ids = np.load(f'{params.apl["path2save"]}/NewsIds.npy')
    user_clusters = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    communities_topic_interests = communities_topic_interest(user_clusters, news_topics)
    return recommend(communities_topic_interests, news_topics, news_ids, topK)

=======
    print('Top Recommendation shape: ', TopRecommendations.shape)

def main(NewsTopics, topK=10):
    NewsIds = np.load(f'{params.apl["path2save"]}/NewsIds.npy')
    UserClusters = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    CommunitiesTopicInterests = CommunitiesTopicInterest(UserClusters, NewsTopics)
    TopRecommendations = Recommend(CommunitiesTopicInterests, NewsTopics, NewsIds, topK)
    return TopRecommendations
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c

