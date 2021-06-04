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
    plt.savefig(f'../output/{params.evl["RunId"]}/evl/{savename}.jpg')
    plt.close()
    return comm_news

# def ChangeLoc():
#     run_list = glob.glob('../output/2021*')
#     print(run_list[-1])
#     os.chdir(run_list[-1])
#
# def LogFile():
#     file_handler = logging.FileHandler("../logfile.log")
#     logger = logging.getLogger()
#     logger.addHandler(file_handler)
#     logger.setLevel(logging.ERROR)
#     return logger

def main(topK = 10):
    # logger = LogFile()
    cmn.logger.info("\nNewsRecommendation2.py:\n")

    # LDA Model and News Topics Loading
    model_name = glob.glob(f'../output/{params.evl["RunId"]}/tml/*.model')[0].split("/")[-1]
    print(os.getcwd())
    print(model_name)
    num_topics = int(model_name.split('.')[0].split('_')[1].split('t')[0])
    print(num_topics)
    GenMal = model_name.split('\\')[-1].split('_')[0]
    if GenMal == 'gensim':
        ldaModel = gensim.models.ldamodel.LdaModel.load(f'../output/{params.evl["RunId"]}/tml/{model_name}')
        print('Lda Model Loaded (Gensim)')
    elif GenMal == 'mallet':
        ldaModel = gensim.models.wrappers.LdaMallet.load(f'../output/{params.evl["RunId"]}/tml/{model_name}')
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        print('Lda Model Loaded (Mallet)')
    else:
        print('Wrong Library!')
    NewsTopics = ldaModel.load(f'../output/{params.evl["RunId"]}/evl/NewsTopics.mm')

    # User Clusters loading
    UserClusters = np.load(f'../output/{params.evl["RunId"]}/uml/UserClusters.npy')

    # Users Topic Interest loading
    UsersTopicInterestsList = glob.glob(f'../output/{params.evl["RunId"]}/uml/Day*UsersTopicInterests.npy')
    print(UsersTopicInterestsList[-1])
    LastUTI = np.load(UsersTopicInterestsList[-1])
    CommunitiesTopicInterests = []
    ClusterNumbers = []
    for UC in range(UserClusters.min(), UserClusters.max()+1):
        UsersinCluster = np.where(UserClusters == UC)[0]
        if len(UsersinCluster) == 1:
            break
        TopicInterestSum = np.zeros(num_topics)
        for user in UsersinCluster:
            TopicInterestSum += LastUTI[user]
        CommunitiesTopicInterests.append(TopicInterestSum)
        ClusterNumbers.append(UC)
    print('len:',len(CommunitiesTopicInterests))
    CommunitiesTopicInterests = np.asarray(CommunitiesTopicInterests)
    np.save(f'../output/{params.evl["RunId"]}/evl/CommunitiesTopicInterests.npy', CommunitiesTopicInterests)
    np.save(f'../output/{params.evl["RunId"]}/evl/ClusterNumbers.npy', ClusterNumbers)
    News = np.zeros((len(NewsTopics), num_topics))
    for NT in range(len(NewsTopics)):
        NewsVector = np.asarray(NewsTopics[NT])
        NewsVector_temp = np.zeros(num_topics)
        for topic in NewsVector:
            NewsVector_temp[int(topic[0])] = topic[1]
        News[NT] = NewsVector_temp
    # np.save('../NewsTopicInterests.npy', News)
    RecommendationTable = np.matmul(CommunitiesTopicInterests, News.T)
    TopRecommendations = np.zeros((len(CommunitiesTopicInterests), topK))
    for r in range(len(RecommendationTable)):
        NewsScores = RecommendationTable[r]
        fifthScore = np.partition(NewsScores.flatten(), -topK)[-topK]
        RecommendationCandidates = np.where(NewsScores >= fifthScore)[0]
        RecommendationScores = NewsScores[RecommendationCandidates]
        inds = np.flip(RecommendationScores.argsort())
        Recommendations_sorted = RecommendationCandidates[inds]
        TopRecommendations[r] = Recommendations_sorted[:topK]


    # RecommendationTable = RecommendationTable.T
    # RecommendationTable_expanded = RecommendationTable_expanded.T
    RecommendationTableAnalyzer(RecommendationTable, 'NRN', 'CommunityPerNewsNumbers')
    RecommendationTableAnalyzer(RecommendationTable, 'CRN', 'NewsPerCommunityNumbers')
    np.save(f'../output/{params.evl["RunId"]}/evl/TopRecommendations.npy', TopRecommendations)
    cmn.logger.info("Shape of TopRecommendations: "+str(TopRecommendations.shape))
    # np.save('../RecommendationTable.npy', RecommendationTable)
    # np.save('../RecommendationTableExpanded.npy', RecommendationTable_expanded)

