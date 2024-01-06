import numpy as np
import pandas as pd


import params
from tml import TopicModeling as tm
from cmn import Common as cmn
from dal import DataPreparation as DP

def main(news_table, dictionary, lda_model):
    news_table.dropna(subset=[params.apl["textTitle"]], inplace=True)
    # news_table = DP.reassign_id(news_table,'NewsId')
    if params.dal['preProcessing']:
        news_table['Tokens'] = DP.preprocess_tweets(news_table[params.apl['textTitle']])
    else:
        news_table['Tokens'] = news_table['textTitle'].str.split()

    cols_to_drop = ['ExpandedUrl', 'ShortUrl','DisplayUrl','SourceUrl','Text','Title','Description']
    news_table.drop(cols_to_drop, axis=1, inplace=True)
    news_table['TopicInterests'] = pd.Series
    news_table.astype(object)
    for index, row in news_table.iterrows():
        news_table.loc[index, 'TopicInterests'] = str(list(tm.doc2topics(lda_model, dictionary.doc2bow(row['Tokens']))))
    cols_to_drop = ['Tokens']
    news_table.drop(cols_to_drop, axis=1, inplace=True)
    news_table.to_csv(f"../output/{params.apl['path2save']}/documents.csv", encoding='utf-8', index=False, header=True)
    return news_table


