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
    NewsIds = np.load(f'../output/{params.evl["RunId"]}/evl/NewsIds.npy')
    # LDA Model and News Topics Loading
    model_name = glob.glob(f'../output/{params.evl["RunId"]}/tml/*.model')[0].split("/")[-1].split('\\')[-1]
    # print(os.getcwd())
    # print(model_name)
    num_topics = int(model_name.split('.')[0].split('_')[1].split('t')[0])
    # print(num_topics)
    GenMal = model_name.split('\\')[-1].split('_')[0]
    # print(GenMal)
    if GenMal == 'gensim':
        ldaModel = gensim.models.ldamodel.LdaModel.load(f'../output/{params.evl["RunId"]}/tml/{model_name}')
        print('Lda Model Loaded (Gensim)')
    elif GenMal == 'mallet':
        ldaModel = gensim.models.wrappers.LdaMallet.load(f'../output/{params.evl["RunId"]}/tml/{model_name}')
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        print('Lda Model Loaded (Mallet)')
    else:
        print('Wrong Library!')
    #NewsTopics = ldaModel.load(f'../output/{params.evl["RunId"]}/evl/NewsTopics.mm')
    NewsTopics = np.load(f'../output/{params.evl["RunId"]}/evl/NewsTopics.npy')
    # topK = len(NewsTopics)
    # User Clusters loading
    UserClusters = np.load(f'../output/{params.evl["RunId"]}/uml/UserClusters.npy')

    # Users Topic Interest loading
    UsersTopicInterestsList = glob.glob(f'../output/{params.evl["RunId"]}/uml/Day*UsersTopicInterests.npy')
    # print(UsersTopicInterestsList[-1])
    LastUTI = np.load(UsersTopicInterestsList[-1])
    CommunitiesTopicInterests = []
    ClusterNumbers = []
    for UC in range(UserClusters.min(), UserClusters.max()+1):
        UsersinCluster = np.where(UserClusters == UC)[0]
        # print(UsersinCluster)
        if len(UsersinCluster) < 10:
            break
        TopicInterestSum = np.zeros(num_topics)
        for user in UsersinCluster:
            TopicInterestSum += LastUTI[user]
        CommunitiesTopicInterests.append(TopicInterestSum)
        ClusterNumbers.append(UC)
    print('len CommunitiesTopicInterests:', len(CommunitiesTopicInterests))
    CommunitiesTopicInterests = np.asarray(CommunitiesTopicInterests)
    np.save(f'../output/{params.evl["RunId"]}/evl/CommunitiesTopicInterests.npy', CommunitiesTopicInterests)
    cmn.save2excel(CommunitiesTopicInterests, 'evl/CommunitiesTopicInterests')
    np.save(f'../output/{params.evl["RunId"]}/evl/ClusterNumbers.npy', ClusterNumbers)
    cmn.save2excel(ClusterNumbers, 'evl/ClusterNumbers')
    News = np.zeros((len(NewsTopics), num_topics))
    for NT in range(len(NewsTopics)):
        NewsVector = np.asarray(NewsTopics[NT])
        NewsVector_temp = np.zeros(num_topics)
        counter = 0
        for topic in NewsVector:
            NewsVector_temp[counter] = topic
            counter += 1
        News[NT] = NewsVector_temp
    # np.save('../NewsTopicInterests.npy', News)
    RecommendationTable = np.matmul(CommunitiesTopicInterests, News.T)
    cmn.save2excel(RecommendationTable, 'evl/RecommendationTable')
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
    cmn.save2excel(RecommendationTable[r], 'evl/NR_RecommendationTable_r')
    cmn.save2excel(NewsScores, 'evl/NR_NewsScores')
    cmn.save2excel(RecommendationCandidates, 'evl/NR_RecommendationCandidates')
    cmn.save2excel(RecommendationScores, 'evl/NR_RecommendationScores')
    cmn.save2excel(inds, 'evl/NR_inds')
    cmn.save2excel(Recommendations_sorted, 'evl/NR_Recommendations_sorted')
    cmn.save2excel(TopRecommendations[r], 'evl/NR_TopRecommendations_r')


    # RecommendationTable = RecommendationTable.T
    # RecommendationTable_expanded = RecommendationTable_expanded.T
    RecommendationTableAnalyzer(RecommendationTable, 'NRN', 'CommunityPerNewsNumbers')
    RecommendationTableAnalyzer(RecommendationTable, 'CRN', 'NewsPerCommunityNumbers')
    np.save(f'../output/{params.evl["RunId"]}/evl/TopRecommendations.npy', TopRecommendations)

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
    cmn.logger.info("Shape of TopRecommendations: "+str(TopRecommendations.shape))
    # np.save('../RecommendationTable.npy', RecommendationTable)
    # np.save('../RecommendationTableExpanded.npy', RecommendationTable_expanded)

