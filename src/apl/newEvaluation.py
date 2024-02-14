import pandas as pd
import pytrec_eval
import json
import os
from tqdm import tqdm

import params

def main(news_table, user_community_recommendation, user_recommendation):
    if params.evl["recommendationType"] == 'community_recommendation': recommendation = user_community_recommendation
    elif params.evl["recommendationType"] == 'user_recommendation': recommendation = user_recommendation
    else: print('bad recommendation type')
    if not os.path.isdir(f'{params.apl["path2save"]}/evl'): os.makedirs(f'{params.apl["path2save"]}/evl')
    URLs = pd.read_csv(f'{params.dal["path"]}/TweetEntities.csv', encoding='utf-8')
    URLs = URLs[URLs['EntityTypeCode'] == 2].copy()
    columns_to_drop = ['Text', 'StartIndex', 'EndIndex', 'EntityTypeCode', 'Name', 'ScreenName', 'MediaUrl',
                       'MediaUrlHttps', 'MediaType', 'MediaSize']
    URLs.drop(columns=columns_to_drop, inplace=True)
    tweets = pd.read_csv(f'{params.dal["path"]}/Tweets.csv', encoding='utf-8')
    URLs = pd.merge(URLs, tweets[['Id', 'UserId']], left_on='TweetId', right_on='Id', how='left')
    URLs.drop(columns=['Id_y', 'UserOrMediaId'], inplace=True)
    URLs = URLs[URLs['UserId'].isin(recommendation['UserId'])]
    recommendation = recommendation[recommendation['UserId'].isin(URLs['UserId'])]
    tweet_user_news_table = pd.merge(URLs, news_table[['NewsId', 'ExpandedUrl', 'ShortUrl', 'DisplayUrl', 'SourceUrl']],
                         left_on=['Url', 'ExpandedUrl', 'DisplayUrl'],
                         right_on=['ExpandedUrl', 'ShortUrl', 'DisplayUrl'],
                         how='left')
    tweet_user_news_table = tweet_user_news_table[tweet_user_news_table['NewsId'].notna()]
    tweet_user_news_table['NewsId'] = tweet_user_news_table['NewsId'].astype('Int64')
    tweet_user_news_table.to_csv(f'{params.apl["path2save"]}/tweet_user_news_table.csv', index=False)
    URLs.to_csv(f'{params.dal["path"]}/TweetEntities_clean.csv', index=False)
    tweet_user_news_table = tweet_user_news_table[tweet_user_news_table['UserId'].isin(recommendation['UserId'])]
    recommendation = recommendation[recommendation['UserId'].isin(tweet_user_news_table['UserId'])]
    trec_mentions = dataframe_to_trec(tweet_user_news_table)
    trec_recommendations = dataframe_to_trec_recommendations(recommendation)
    pytrec_result = pytrec_eval_run(trec_mentions, trec_recommendations)
    # pytrec_result = pytrec_eval_run(trec_recommendations, trec_mentions)
    return pytrec_result


def dataframe_to_trec(df, userIdName='UserId', NewsIdName='NewsId'):
    trec_data = {}
    for index, row in tqdm(df.iterrows()):
        user_id = str(row[userIdName])
        news_id = str(row[NewsIdName])
        relevance = 1
        if user_id not in trec_data: trec_data[user_id] = {news_id: relevance}
        else: trec_data[user_id][news_id] = relevance
    return trec_data

def dataframe_to_trec_recommendations(df, userIdName='UserId'):
    trec_data = {}
    for index, row in tqdm(df.iterrows()):
        user_id = str(row[userIdName])
        top_news_columns = [f'TopNews_{i}' for i in range(1, params.evl['topK']+1)]
        recommendations = [str(row[col]) for col in top_news_columns if pd.notna(row[col])]
        relevance = 1
        if user_id not in trec_data:
            trec_data[user_id] = {}
        trec_data[user_id].update({news_id: relevance for news_id in recommendations})
    return trec_data

def pytrec_eval_run(qrel, run):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, params.evl['extrinsicEvaluationMetrics'])
    pred_eval = evaluator.evaluate(run)
    with open(f'{params.apl["path2save"]}/evl/pred.eval.txt', 'w') as outfile:
        json.dump(pred_eval, outfile)
    pred_eval = pd.DataFrame(pred_eval).T
    pred_eval.to_csv(f'{params.apl["path2save"]}/evl/pred.eval.csv')
    mean = pred_eval.mean(axis=0, skipna=True)
    mean.to_csv(f'{params.apl["path2save"]}/evl/pred.eval.mean.csv', index_label="metric", header=["score"])
    return pred_eval