import os
import glob
import gensim
import numpy as np
import matplotlib.pyplot as plt
import logging
import params
from cmn import Common as cmn

def RecommendationTableAnalyzer(RT, test, savename):
    if test == 'CRN': # 'CRN' = 'community recommendations number'
        pass
    elif test == 'NRN': # 'news recommendations number'
        RT = RT.T
    comm_news = []
    for com in RT:
        comm_news.append(com.sum())
    if test == 'NRN':
        comm_news.sort()
    plt.plot(range(len(comm_news)), comm_news)
    plt.savefig(f'{params.apl["path2save"]}/{savename}.jpg')
    plt.close()
    return comm_news


def main(NewsTopics, topK=10):
    NewsIds = np.load(f'{params.apl["path2save"]}/NewsIds.npy')
    UserClusters = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    CommunitiesTopicInterests = CommunitiesTopicInterest(UserClusters, NewsTopics)
    TopRecommendations = recom(CommunitiesTopicInterests, NewsTopics, NewsIds, topK)
    return TopRecommendations


def CommunitiesTopicInterest(UserClusters, NewsTopics):
    num_topics = NewsTopics.shape[1]
    UsersTopicInterestsList = sorted(glob.glob(f'{params.uml["path2save"]}/Day*UsersTopicInterests.npy'))
    LastUTI = np.load(UsersTopicInterestsList[-1])
    CommunitiesTopicInterests = []
    ClusterNumbers = []
    for UC in range(UserClusters.min(), UserClusters.max()+1):
        UsersinCluster = np.where(UserClusters == UC)[0]
        if len(UsersinCluster) < params.cpl["min_size"]:
            break
        TopicInterestSum = np.zeros(num_topics)
        for user in UsersinCluster:
            TopicInterestSum += LastUTI[user]
        CommunitiesTopicInterests.append(TopicInterestSum)
        ClusterNumbers.append(UC)
    print('len CommunitiesTopicInterests:', len(CommunitiesTopicInterests))
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



def recom(CommunitiesTopicInterests, News, NewsIds, topK):
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
    b = set(a)
    print(len(a))
    print(len(b))
    print(len(a)/len(b))
    duplicate = False
    for i in TopRecommendations:
        if len(i) != len(set(i)):
            duplicate = True
            print(i)
    if not duplicate:
        print('All rows has distinct news Ids.', len(i))
    print('Top Recommendation shape: ', TopRecommendations.shape)


