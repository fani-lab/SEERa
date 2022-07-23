import numpy as np
import glob
import os
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from cmn import Common as cmn
from application import NewsTopicExtraction as NTE, NewsRecommendation as NR, PytrecEvaluation as PyEval
import params
import pickle
import sklearn.metrics.cluster as CM
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# for pytrec_eval
<<<<<<< HEAD
def dictonary_generation(top_recommendations, mentions):
    Recommendation = {}
    for c in range(len(top_recommendations)):
        Recommendation['u' + str(c + 1)] = {}
        comm = top_recommendations[c]
        for n in range(len(comm)):
            Recommendation['u'+str(c+1)]['n'+str(int(comm[n]))] = 1
    Mention = {}
    for c in range(len(mentions)):
        Mention['u' + str(c + 1)] = {}
        comm = mentions[c]
=======
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
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
        for n in range(len(comm)):
            Mention['u' + str(c + 1)]['n' + str(int(comm[n]))] = 1
    return  Recommendation, Mention

def userMentions(day_before,end_date):
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
    print('Connection Created')
    cursor = cnx.cursor()
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=day_before)

    sqlScript = '''SELECT TweetId, NewsId, ExpandedUrl, UserId, CreationTimeStamp FROM 
    (SELECT TweetId, News.Id AS NewsId, ExpandedUrl FROM TweetEntities INNER JOIN News on
    ExpandedUrl = News.Url) AS T
    INNER JOIN
    (SELECT Id AS Tid, UserId, CreationTimeStamp  FROM Tweets WHERE UserId != -1) AS T2 on T.TweetId = T2.Tid'''# WHERE length(ExpandedUrl)>40'''
    print(day)
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    print('Connection Closed')
    return table

#########################################################################################################
##########################################################################################################
def userMentionsCSV():
    # creating tweetentities dataframe from tweetentities.csv
    tweetentities = pd.read_csv(r'../../data/tweetentities_N20000.csv', sep=';', encoding='utf-8')
    # renaming Id to NewsId
    tweetentities.rename(columns={"Id": "NewsId"}, inplace=True)

    tweets = pd.read_csv(r'../../data/tweets_N20000.csv', sep=';', encoding='utf-8')  # creating tweets dataframe from csv
    tweets.rename(columns={"Id": "TweetId"}, inplace=True)  # remaining Id in tweets dataframe to TweetsId
    tweets = tweets[tweets.UserId != -1]  # remove user ids with -1 value

    # drop tables we dont need from tweetentities dataframe
    tweetentities = tweetentities.drop(columns=["Text","StartIndex","EndIndex","EntityTypeCode","Url",
                                                "DisplayUrl","Name","ScreenName","UserOrMediaId","MediaUrl",
                                                "MediaUrlHttps","MediaType","MediaSize"])

    # drop tables we dont need from tweets dataframe
    tweets = tweets.drop(columns=["Text","ModificationTimestamp"])

    tweetentities = pd.merge(tweetentities, tweets, on="TweetId") # merge the tables on TweetId
    tweetentities = tweetentities.dropna(subset=["ExpandedUrl"]) # drop rows with NaN in ExpandableUrl

    # print(tweetentities.to_string()) # for testing table output

    return tweetentities
#########################################################################################################
#########################################################################################################
def main(RunId, path2_save_evl):
    os.chdir('../../../')
    if not os.path.isdir(path2_save_evl): os.makedirs(path2_save_evl)
    NTE.main()
    NR.main(topK=params.evl['TopK'])
    cmn.logger.info("\nModelEvaluation.py:\n")
    All_Users = np.load(f'../output/{RunId}/uml/AllUsers.npy')
    end_date = params.uml['end']
    # end_date = np.load(f'../output/{RunId}/uml/end_date.npy', allow_pickle=True)
    TopRecommendations_clusters = np.load(f'../output/{RunId}/evl/TopRecommendations.npy')
    cmn.save2excel(TopRecommendations_clusters, 'evl/TopRecommendations_clusters')
    UC = np.load(f'../output/{RunId}/uml/UserClusters.npy')
    TopRecommendations_Users = np.zeros((UC.shape[0], TopRecommendations_clusters.shape[1]))
    for i in range(TopRecommendations_clusters.shape[0]):
        indices = np.where(UC == i)[0]
        TopRecommendations_Users[indices] = TopRecommendations_clusters[i]
    # cmn.save2excel(TopRecommendations_Users, 'evl/TopRecommendations_Users')
    end_date = pd.Timestamp(str(end_date))
    daybefore = 0
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=daybefore)
    cmn.logger.critical("Selected date for evaluation: "+str(day.date()))
    tbl = userMentions(daybefore, end_date)
    cmn.save2excel(tbl, 'evl/userMentions')
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

    cmn.save2excel(TopRecommendations_Users, 'evl/TopRecommendations_Users')
    cmn.save2excel(Mentions_user, 'evl/Mentions_user')
<<<<<<< HEAD
    r_user, m_user = dictonary_generation(TopRecommendations_Users, Mentions_user)
=======
    r_user, m_user = DictonaryGeneration(TopRecommendations_Users, Mentions_user)
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c

    cmn.save2excel(tbl, 'evl/userMentions')
    save_obj(r_user, f'../output/{RunId}/evl/RecommendedNews_UserBased')
    save_obj(m_user, f'../output/{RunId}/evl/MentionedNews_UserBased')
    clusters = []
    userCounts = []
    for uc in range(1, UC.max()):
        UsersinCluster = np.where(UC == uc)[0]
        if len(UsersinCluster) < 10:
            break
        clusters.append(uc)
        userCounts.append(len(UsersinCluster))
    plt.plot(clusters, userCounts)
    plt.savefig(f'../output/{RunId}/uml/UsersInCluster.jpg')
    pytrec_result1 = PyEval.main(r_user, m_user)
    # pytrec_result2 = PyEval.main2(TopRecommendations_Users, Mentions_user)
    return pytrec_result1#, pytrec_result2

<<<<<<< HEAD
def intrinsic_evaluation(Communities, GoldenStandard, EvaluationMetrics=params.evl['evaluationMetrics']):
=======
def intrinsic_evaluation(Communities, GoldenStandard, EvaluationMetrics=params.evl['EvaluationMetrics']):
>>>>>>> 54bc47755b58cb9863c9bd516cbac720613f723c
    results = []
    for m in EvaluationMetrics:
        if m == 'adjusted_rand':
            results.append(('adjusted_rand_score', CM.adjusted_rand_score(GoldenStandard, Communities)))
        elif m == 'completeness':
            results.append(('completeness_score', CM.completeness_score(GoldenStandard, Communities)))
        elif m == 'homogeneity':
            results.append(('homogeneity_score', CM.homogeneity_score(GoldenStandard, Communities)))
        elif m == 'rand':
            results.append(('rand_score', CM.rand_score(GoldenStandard, Communities)))
        elif m == 'v_measure':
            results.append(('v_measure_score', CM.v_measure_score(GoldenStandard, Communities)))
        elif m == 'normalized_mutual_info' or m == 'NMI':
            results.append(('normalized_mutual_info_score', CM.normalized_mutual_info_score(GoldenStandard, Communities)))
        elif m == 'adjusted_mutual_info' or m == 'AMI':
            results.append(('adjusted_mutual_info_score', CM.adjusted_mutual_info_score(GoldenStandard, Communities)))
        elif m == 'mutual_info' or m == 'MI':
            results.append(('mutual_info_score', CM.mutual_info_score(GoldenStandard, Communities)))
        elif m == 'fowlkes_mallows' or m == 'FMI':
            results.append(('fowlkes_mallows_score', CM.fowlkes_mallows_score(GoldenStandard, Communities)))
        else:
            print('Wrong Clustering Metric!')
            continue
    return results

