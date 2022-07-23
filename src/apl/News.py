import pandas as pd
import os.path
import numpy as np

import params
from cmn import Common as cmn
from apl import NewsCrawler as NC
from apl import NewsTopicExtraction as NTE
from apl import NewsRecommendation as NR
from apl import ModelEvaluation as ME


def stats(news):
    file_object = open(f"{params.apl['path2save']}/NewsStat.txt", 'a')
    #news = pd.read_csv('News.csv')
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
    if not os.path.isdir(params.apl["path2save"]): os.makedirs(params.apl["path2save"])

    news_path = f'{params.apl["path2read"]}/News.csv'
    try:
        cmn.logger.info(f"Loading news articles ...")
        news_table = pd.read_csv(news_path)
    except:
        cmn.logger.info(f"News articles do not exist! Crawling news articles ...")
        NC.news_crawler(news_path)
        stats(news_table)
        news_table = pd.read_csv(news_path)

    cmn.logger.info(f"Inferring news articles' topics ...")
    try:
        news_topics = np.load(f'{params.apl["path2save"]}/NewsTopics.npy')
    except:
        NTE.main(news_table)
        news_topics = np.load(f'{params.apl["path2save"]}/NewsTopics.npy')

    cmn.logger.info(f"Recommending news articles to future communities ...")
    nrr = NR.main(news_topics, params.apl['topK'])

    cmn.logger.info(f"Evaluating recommended news articles ...")
    me = ME.main()



    return nrr
