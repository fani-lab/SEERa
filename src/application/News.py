import pandas as pd
import os.path
from application import NewsCrawler as NC
from application import NewsTopicExtraction as NTE
from application import NewsRecommendation as NR
import numpy as np
from cmn import Common as cmn
import params

def main():
    if not os.path.isdir(params.apl["path2save"]): os.makedirs(params.apl["path2save"])

    NewsPath = f'{params.apl["path2read"]}/News.csv'
    try:
        NewsTable = pd.read_csv(NewsPath)
        #cmn.logger.info(f"Crawling news articles ...")
        #cmn.logger.info(f"News CSV file is saved on {NewsPath}")
    except:
        NC.NewsCrawler(NewsPath)
        NewsTable = pd.read_csv(NewsPath)

    print(NewsTable.shape)

    # NEWS TOPIC EXTRACTION:
    try:
        NewsTopics = np.load(f'{params.apl["path2save"]}/NewsTopics.npy')
    except:
        NTE.main(NewsTable, newsstat=True)
        NewsTopics = np.load(f'{params.apl["path2save"]}/NewsTopics.npy')


    NRR = NR.main(NewsTopics, params.apl['TopK'])
    return NRR


#a = main()