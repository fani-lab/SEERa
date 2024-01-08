import pandas as pd
import os.path

import params
from cmn import Common as cmn
from apl import NewsCrawler as NC
from apl import NewsTopicExtraction as NTE
from apl import NewsRecommendation as NR
from apl import newEvaluation as NE

def stats(news):
    save_path = params.apl['path2save']
    with open(f"{save_path}/NewsStat.txt", 'a') as file_object:
        texts = news.Text.dropna()
        titles = news.Title.dropna()
        desc = news.Description.dropna()
        file_object.write(f'Available texts: {len(texts)}\n')
        file_object.write(f'Available titles: {len(titles)}\n')
        file_object.write(f'Available descriptions: {len(desc)}\n')
        avg_word_length = lambda col: sum(len(text.split()) for text in col) // len(col)
        file_object.write(f'Average texts length: {avg_word_length(texts)} words.\n')
        file_object.write(f'Average titles length: {avg_word_length(titles)} words.\n')
        file_object.write(f'Average descriptions length: {avg_word_length(desc)} words.\n')

def main(user_final_interests, user_clusters, dictionary, lda_model):
    if not os.path.isdir(params.apl["path2save"]): os.makedirs(params.apl["path2save"])

    news_path = f'{params.dal["path"]}/News.csv'
    tweet_entities_path = f'{params.dal["path"]}/TweetEntities.csv'
    try:
        cmn.logger.info(f"Loading news articles ...")
        news_table = pd.read_csv(news_path)
        if params.apl['stat']: stats(news_table)
    except:
        cmn.logger.info(f"News articles do not exist! Crawling news articles ...")
        NC.news_crawler(news_path, tweet_entities_path)
        stats(news_table)
        news_table = pd.read_csv(news_path)
    try:
        cmn.logger.info(f"Loading news articles' topics ...")
        from ast import literal_eval
        news_table = pd.read_csv(f"../output/{params.apl['path2save']}/documents.csv", converters={"TopicInterests": literal_eval})
    except:
        cmn.logger.info(f"Loading news articles' topics failed! Inferring news articles' topics ...")
        news_table = NTE.main(news_table, dictionary, lda_model)
    try:
        cmn.logger.info(f"Loading news article recommendations ...")
        community_recommendation = pd.read_csv(f"../output/{params.apl['path2save']}/community_recommendations.csv")
        user_community_recommendation = pd.read_csv(f"../output/{params.apl['path2save']}/user_community_recommendations.csv")
        user_recommendation = pd.read_csv(f"../output/{params.apl['path2save']}/user_recommendations.csv")
    except:
        cmn.logger.info(f"Loading news article recommendations failed! Recommending news articles to future communities ...")
        community_recommendation, user_community_recommendation, user_recommendation = NR.main(user_clusters, user_final_interests, news_table)
    try:
        evaluation_results =
    except:
        cmn.logger.info(f"Evaluating recommended news articles ...")
        evaluation_results = NE.main(news_table, user_community_recommendation, user_recommendation)
    return evaluation_results
