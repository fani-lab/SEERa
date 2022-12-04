import pandas as pd
import os.path
from os.path import exists

import Params
from cmn import Common as cmn
from apl import NewsCrawler as NC
from apl import NewsTopicExtraction as NTE
from apl import NewsRecommendation as NR
from apl import ModelEvaluation as ME


def stats(news):
    file_object = open(f"{Params.apl['path2save']}/NewsStat.txt", 'a')
    texts = news.Text.dropna()
    titles = news.Title.dropna()
    desc = news.Description.dropna()
    cmn.logger.info("Available texts: ", len(texts))
    file_object.write(f'Available texts: {len(texts)}\n')
    cmn.logger.info("Available titles: ", len(titles))
    file_object.write(f'Available titles: {len(titles)}\n')
    cmn.logger.info("Available descriptions: ", len(desc))
    file_object.write(f'Available descriptions: {len(desc)}\n')

    sum_text = 0
    for i in range(len(texts)):
        sum_text += len(texts.values[i].split())
    text_avg = sum_text//i #Hossein: What's this? Soroush: Avg number of words for all text blocks.
    cmn.logger.info("Average texts length: ", text_avg)
    file_object.write(f'Average texts length: {text_avg} words.\n')

    sum_title = 0
    for i in range(len(titles)): sum_title += len(titles.values[i].split())
    title_avg = sum_title//i
    cmn.logger.info("Average titles length: ", title_avg)
    file_object.write(f'Average titles length: {title_avg} words.\n')

    sum_desc = 0
    for i in range(len(desc)): sum_desc += len(desc.values[i].split())
    desc_avg = sum_desc//i
    print("Average descriptions length: ", desc_avg)
    file_object.write(f'Average descriptions length: {desc_avg} words.\n')
    file_object.close()


def main():
    if not os.path.isdir(Params.apl["path2save"]): os.makedirs(Params.apl["path2save"])

    news_path = f'{Params.dal["path"]}/News.csv'
    tweet_entities_path = f'{Params.dal["path"]}/TweetEntities.csv'
    try:
        cmn.logger.info(f"6.1 Loading news articles ...")
        news_table = pd.read_csv(news_path)
    except:
        cmn.logger.info(f"6.1 News articles do not exist! Crawling news articles ...")
        NC.news_crawler(news_path, tweet_entities_path)
        news_table = pd.read_csv(news_path)
        stats(news_table)

    cmn.logger.info(f"6.2 Inferring news articles' topics ...")
    try: news_topics = pd.read_pickle(f'{Params.apl["path2save"]}/NewsTopics.pkl')
    except (FileNotFoundError, EOFError) as e: news_topics = NTE.main(news_table)

    cmn.logger.info(f"6.3 Recommending news articles to future communities ...")
    try: final_recommendation = pd.read_pickle(f'{Params.apl["path2save"]}/TopRecommendationsUser.pkl')
    except (FileNotFoundError, EOFError) as e: final_recommendation = NR.main(news_topics, Params.apl['topK'])

    end_date = pd.Timestamp(str(Params.dal['end']))
    day_before = 0
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=day_before)
    cmn.logger.info(f"6.4 Evaluating recommended news articles on future time interval {str(day.date())}...")
    me = ME.main(final_recommendation)



    return me
