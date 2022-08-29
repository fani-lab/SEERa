import numpy as np
import pandas as pd
import pytrec_eval
import matplotlib.pyplot as plt
import os
import json
import datetime
from cmn import Common as cmn
import pickle
import sklearn.metrics.cluster as CM

import Params


def pytrec_eval_run(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, Params.evl['extrinsicEvaluationMetrics'])
    pred_eval = evaluator.evaluate(run)
    pred_eval = pd.DataFrame(pred_eval).T
    pred_eval.to_csv(f'{Params.apl["path2save"]}/evl/Pred.Eval.csv')
    mean = pred_eval.mean(axis=0, skipna=True)
    mean.to_csv(f'{Params.apl["path2save"]}/evl/Pred.Eval.Mean.csv', index_label="metric", header=["score"])
    return pred_eval


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def dictionary_generation(top_recommendations, mentions):
    recommendation = {}
    for user in top_recommendations:
        recoms = top_recommendations[user]
        recommendation[str(user)] = {}
        for n in range(len(recoms)):
            recommendation[str(user)][str(int(recoms[n]))] = 1

    mention = {}
    for user in mentions:
        recoms = mentions[user]
        mention[str(user)] = {}
        for n in range(len(recoms)):
            mention[str(user)][str(int(recoms[n]))] = 1

    return recommendation, mention


def user_mentions():
    news = pd.read_csv(f'{Params.dal["path"]}/News.csv')
    tweet_entities = pd.read_csv(f'{Params.dal["path"]}/TweetEntities.csv')
    tweets = pd.read_csv(f'{Params.dal["path"]}/Tweets.csv')
    tweet_entities_with_tweetid = tweet_entities[tweet_entities['TweetId'].notna()]
    tweet_entities_with_tweetid_and_url = tweet_entities_with_tweetid[tweet_entities_with_tweetid['ExpandedUrl'].notna()]
    users_news = {}
    for index, row in tweet_entities_with_tweetid_and_url.iterrows():
        row_date = tweets.loc[tweets['Id'] == row['TweetId']]['CreationTimestamp'].values[0].split()[0]
        is_last = (datetime.datetime.strptime('2010-12-04', '%Y-%m-%d') - datetime.datetime.strptime(row_date,'%Y-%m-%d')).days < 1  # May need changes for new news dataset
        uid = tweets.loc[tweets['Id'] == row['TweetId']]['UserId'].values[0]
        if pd.notna(uid) and is_last:
            try:
                news_id = news[news['ExpandedUrl'] == row['ExpandedUrl']]['NewsId'].values[0]
                users_news.setdefault(uid, []).append(news_id)
            except:
                pass
    return users_news


def intrinsic_evaluation(communities, golden_standard, evaluation_metrics=Params.evl['intrinsicEvaluationMetrics']):
    results = []
    for m in evaluation_metrics:
        if m == 'adjusted_rand':
            results.append(('adjusted_rand_score', CM.adjusted_rand_score(golden_standard, communities)))
        elif m == 'completeness':
            results.append(('completeness_score', CM.completeness_score(golden_standard, communities)))
        elif m == 'homogeneity':
            results.append(('homogeneity_score', CM.homogeneity_score(golden_standard, communities)))
        elif m == 'rand':
            results.append(('rand_score', CM.rand_score(golden_standard, communities)))
        elif m == 'v_measure':
            results.append(('v_measure_score', CM.v_measure_score(golden_standard, communities)))
        elif m == 'normalized_mutual_info' or m == 'NMI':
            results.append(('normalized_mutual_info_score', CM.normalized_mutual_info_score(golden_standard, communities)))
        elif m == 'adjusted_mutual_info' or m == 'AMI':
            results.append(('adjusted_mutual_info_score', CM.adjusted_mutual_info_score(golden_standard, communities)))
        elif m == 'mutual_info' or m == 'MI':
            results.append(('mutual_info_score', CM.mutual_info_score(golden_standard, communities)))
        elif m == 'fowlkes_mallows' or m == 'FMI':
            results.append(('fowlkes_mallows_score', CM.fowlkes_mallows_score(golden_standard, communities)))
        else:
            print('Wrong Clustering Metric!')
            continue
    return results


def main():
    if not os.path.isdir(f'{Params.apl["path2save"]}/evl'): os.makedirs(f'{Params.apl["path2save"]}/evl')
    if Params.evl['evaluationType'] == "Intrinsic":
        user_communities = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
        golden_standard = np.load(f'{Params.dal["path"]}/GoldenStandard.npy')
        scores = np.asarray(intrinsic_evaluation(user_communities, golden_standard))
        np.save(f'{Params.apl["path2save"]}/evl/IntrinsicEvaluationResult.npy', scores)
        return scores
    if Params.evl['evaluationType'] == "Extrinsic":
        tbl = user_mentions()
        pd.to_pickle(tbl, f'{Params.apl["path2save"]}/evl/UserMentions.pkl')
        top_recommendation_user = pd.read_pickle(f'{Params.apl["path2save"]}/TopRecommendationsUser.pkl')
        r_user, m_user = dictionary_generation(top_recommendation_user, tbl)
        pytrec_result = pytrec_eval_run(m_user, r_user)
        return pytrec_result
