import numpy as np
import glob
import os
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from cmn import Common as cmn
from evl import NewsTopicExtraction as NTE, NewsRecommendation as NR#, PytrecEvaluation as PyEval
import params
import pickle


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# for pytrec_eval
def DictonaryGeneration(topRecommendations, Mentions):
    Recommendation = {}
    for c in range(len(topRecommendations)):
        Recommendation['u' + str(c + 1)] = {}
        comm = topRecommendations[c]
        for n in range(len(comm)):
            Recommendation['u'+str(c+1)]['n'+str(int(comm[n]))] = 1
    Mention = {}
    for c in range(len(Mentions)):
        Mention['u' + str(c + 1)] = {}
        comm = Mentions[c]
        for n in range(len(comm)):
            Mention['u' + str(c + 1)]['n' + str(int(comm[n]))] = 1
    return  Recommendation, Mention

def userMentions(day_before,end_date):
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673',
                                  host='localhost',
                                  database='twitter3')
    print('Connection Created')
    cursor = cnx.cursor()
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=day_before)

    sqlScript = '''SELECT TweetId, NewsId, ExpandedUrl, UserId, CreationTimeStamp FROM 
    (SELECT TweetId, news.Id AS NewsId, ExpandedUrl FROM twitter3.tweetentities INNER JOIN twitter3.news on
    ExpandedUrl = news.Url) AS T
    INNER JOIN
    (SELECT Id AS Tid, UserId, CreationTimeStamp  FROM twitter3.tweets WHERE UserId != -1) AS T2 on T.TweetId = T2.Tid'''# WHERE length(ExpandedUrl)>40'''
    print(day)
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    print('Connection Closed')
    return table

# def ChangeLoc():
#     run_list = glob.glob('../output/2021*')
#     print(run_list[-1])
#     os.chdir(run_list[-1]+'/graphs')
#
# def LogFile():
#     file_handler = logging.FileHandler("../logfile.log")
#     logger = logging.getLogger()
#     logger.addHandler(file_handler)
#     logger.setLevel(logging.ERROR)
#     return logger

def main(RunId, path2_save_evl, ):
    if not os.path.isdir(path2_save_evl): os.makedirs(path2_save_evl)
    NTE.main()
    NR.main(topK=params.evl['TopK'])
    cmn.logger.info("\nModelEvaluation.py:\n")
    All_Users = np.load(f'../output/{RunId}/uml/AllUsers.npy')
    end_date = params.uml['end']
    # end_date = np.load(f'../output/{RunId}/uml/end_date.npy', allow_pickle=True)
    TopRecommendations_clusters = np.load(f'../output/{RunId}/evl/TopRecommendations.npy')
    UC = np.load(f'../output/{RunId}/uml/UserClusters.npy')
    TopRecommendations_Users = np.zeros((UC.shape[0], TopRecommendations_clusters.shape[1]))
    for i in range(TopRecommendations_clusters.shape[0]):
        indices = np.where(UC == i)[0]
        TopRecommendations_Users[indices] = TopRecommendations_clusters[i]
    end_date = pd.Timestamp(str(end_date))
    daybefore = 0
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=daybefore)
    cmn.logger.critical("Selected date for evaluation: "+str(day.date()))
    tbl = userMentions(daybefore, end_date)

    Mentions_user = []
    MentionerUsers = []
    MissedUsers = []
    Counter = 0

    for i in range(len(All_Users)):
        Mentions_user.append([])
    for row in tbl: # tid, nid, url, uid, time
        NID, UID = row[1], np.where(All_Users == row[3])
        if len(UID[0]) != 0:
            MentionerUsers.append(UID)
            Mentions_user[UID[0][0]].append(NID)
        else:
            MissedUsers.append(row[3])
        Counter += 1

    MentionNumbers_user = []
    Mentioners = 0
    for i in range(len(Mentions_user)):
        MentionNumbers_user.append(len(Mentions_user[i]))
        if len(Mentions_user[i]) > 0:
            Mentioners += 1


    print('User: Mentions:', sum(MentionNumbers_user), '/', 'Missed Users:', len(MissedUsers), '/', 'Mentioners:', Mentioners, '/', 'All Users:', len(All_Users))
    print('User: total:', Counter, '/', 'sum:', sum(MentionNumbers_user)+len(MissedUsers))

    cmn.logger.critical('\nUser: Mentions:'+ str(sum(MentionNumbers_user))+ ' / Missed Users:'+str(len(MissedUsers))+' / Mentioners:'+ str(Mentioners)+ ' / All Users:'+str(len(All_Users)))
    cmn.logger.critical('User: total:'+str(Counter)+ ' / sum:'+ str(sum(MentionNumbers_user)+len(MissedUsers)))


    r_user, m_user = DictonaryGeneration(TopRecommendations_Users, Mentions_user)
    save_obj(r_user, f'../output/{RunId}/evl/RecommendedNews_UserBased')
    save_obj(m_user, f'../output/{RunId}/evl/MentionedNews_UserBased')
    clusters = []
    userCounts = []
    for uc in range(1, UC.max()):
        UsersinCluster = np.where(UC == uc)[0]
        if len(UsersinCluster) == 1:
            break
        clusters.append(uc)
        userCounts.append(len(UsersinCluster))
    plt.plot(clusters, userCounts)
    plt.savefig(f'../output/{RunId}/uml/UsersInCluster.jpg')
    pytrec_result = PyEval.main(r_user, m_user)
    return pytrec_result