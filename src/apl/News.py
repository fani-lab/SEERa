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

        cmn.logger.info("Available texts:", len(texts))
        file_object.write(f'Available texts: {len(texts)}\n')

        cmn.logger.info("Available titles:", len(titles))
        file_object.write(f'Available titles: {len(titles)}\n')

        cmn.logger.info("Available descriptions:", len(desc))
        file_object.write(f'Available descriptions: {len(desc)}\n')

        avg_word_length = lambda col: sum(len(text.split()) for text in col) // len(col)

        text_avg = avg_word_length(texts)
        cmn.logger.info("Average texts length:", text_avg)
        file_object.write(f'Average texts length: {text_avg} words.\n')

        title_avg = avg_word_length(titles)
        cmn.logger.info("Average titles length:", title_avg)
        file_object.write(f'Average titles length: {title_avg} words.\n')

        desc_avg = avg_word_length(desc)
        cmn.logger.info("Average descriptions length:", desc_avg)
        file_object.write(f'Average descriptions length: {desc_avg} words.\n')

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
    cmn.logger.info(f"Inferring news articles' topics ...")
    try:
        from ast import literal_eval
        news_table = pd.read_csv(f"../output/{params.apl['path2save']}/documents.csv", converters={"TopicInterests": literal_eval})
    except:
        news_table = NTE.main(news_table, dictionary, lda_model)

    cmn.logger.info(f"Recommending news articles to future communities ...")
    community_recommendation, user_community_recommendation, user_recommendation = NR.main(user_clusters, user_final_interests, news_table)

    cmn.logger.info(f"Evaluating recommended news articles ...")
    n = NE.main(news_table, user_community_recommendation, user_recommendation)

    return n
