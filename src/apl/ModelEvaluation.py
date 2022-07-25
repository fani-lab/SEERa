import numpy as np
import glob
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from cmn import Common as cmn
from apl import NewsTopicExtraction as NTE, NewsRecommendation as NR, PytrecEvaluation as PyEval
import params
import pickle
import sklearn.metrics.cluster as CM
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# for pytrec_eval
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
        for n in range(len(comm)):
            Mention['u' + str(c + 1)]['n' + str(int(comm[n]))] = 1
    return Recommendation, Mention


def userMentions():

    tweet_entities = pd.read_csv(f'{params.dal["toyPath"]}/TweetEntities.csv')
    tweets = pd.read_csv(f'{params.dal["toyPath"]}/Tweets.csv')
    ids = tweets.Id.values
    s = []
    for index, row in tweet_entities.iterrows():
        if row['TweetId'] in ids:
            row_date = tweets.loc[tweets['Id'] == row['TweetId']]['CreationTimestamp'].values[0].split()[0]
            if row_date == params.dal['end']:
                s.append(row)
    s = pd.DataFrame(s)
    s = s.dropna(subset=['UserOrMediaId'])
    users = np.unique(s['UserOrMediaId'])
    user_news = {}
    for u in users:
        expanded_url = s.loc[s['UserOrMediaId'] == u]['ExpandedUrl']
        user_news[u] = list(s.loc[s['UserOrMediaId'] == u]['Id'].values)
    return user_news

    # end date (12-04) has 118148 rows in tweetentities.
    # 58503 of them have userId in tweetentities table
    # and 39649 of them can be found in users.npy.
    # 20723 unique users


def main():
    top_recommendation_user = np.load(f'{params.apl["path2save"]}/TopRecommendationsUser.npy')
    #if not os.path.isdir(path2_save_evl): os.makedirs(path2_save_evl)
    #cmn.logger.info("\nModelEvaluation.py:\n")
    #cmn.save2excel(TopRecommendations_clusters, 'evl/TopRecommendations_clusters')
    UC = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    end_date = pd.Timestamp(str(params.dal['end']))
    day_before = 0
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=day_before)
    cmn.logger.info("Selected date for evaluation: "+str(day.date()))
    tbl = userMentions()
    f = open("userMentions.pkl", "wb")
    pickle.dump(tbl, f)
    f.close()
    #cmn.save2excel(tbl, 'evl/userMentions')

    '''
    Mentions_user = []
    MentionerUsers = []
    MissedUsers = []
    Counter = 0

    for i in range(len(all_users)):
        Mentions_user.append([])
    for row in tbl: # tid, nid, url, uid, time
        NID, UID = row[1], np.where(all_users == row[3])
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


    print('User: Mentions:', sum(MentionNumbers_user), '/', 'Missed Users:', len(MissedUsers), '/', 'Mentioners:', Mentioners, '/', 'All Users:', len(all_users))
    print('User: total:', Counter, '/', 'sum:', sum(MentionNumbers_user)+len(MissedUsers))

    #cmn.logger.critical('\nUser: Mentions:'+ str(sum(MentionNumbers_user))+ ' / Missed Users:'+str(len(MissedUsers))+' / Mentioners:'+ str(Mentioners)+ ' / All Users:'+str(len(All_Users)))
    #cmn.logger.critical('User: total:'+str(Counter)+ ' / sum:'+ str(sum(MentionNumbers_user)+len(MissedUsers)))

    #cmn.save2excel(TopRecommendations_Users, 'evl/TopRecommendations_Users')
    #cmn.save2excel(Mentions_user, 'evl/Mentions_user')
    '''
    r_user, m_user = dictonary_generation(top_recommendation_user, tbl)

    #cmn.save2excel(tbl, 'evl/userMentions')
    #save_obj(r_user, f'../output/{RunId}/evl/RecommendedNews_UserBased')
    #save_obj(m_user, f'../output/{RunId}/evl/MentionedNews_UserBased')
    clusters = []
    userCounts = []
    for uc in range(1, UC.max()):
        UsersinCluster = np.where(UC == uc)[0]
        if len(UsersinCluster) < params.cpl['minSize']:
            break
        clusters.append(uc)
        userCounts.append(len(UsersinCluster))
    plt.plot(clusters, userCounts)
    plt.savefig(f'{params.apl["path2save"]}/UsersInCluster.jpg')
    pytrec_result1 = PyEval.main(r_user, m_user)
    # pytrec_result2 = PyEval.main2(TopRecommendations_Users, Mentions_user)
    return pytrec_result1#, pytrec_result2

def intrinsic_evaluation(Communities, GoldenStandard, EvaluationMetrics=params.evl['evaluationMetrics']):
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

