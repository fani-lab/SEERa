import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import glob
import os, sys
import tagme
import params
from tml import TopicModeling as tm
from cmn import Common as cmn
from dal import DataPreparation as DP


def text2tagme(NewsTable, threshold=0.05):
    for i in range(len(NewsTable)):
        text = NewsTable[params.apl["Text_Title"]][i]
        annotations = tagme.annotate(text)
        result = []
        if annotations is not None:
            for keyword in annotations.get_annotations(threshold):
                result.append([keyword.entity_title, keyword.score, NewsTable.index[i]])
    Headers = ['Id', 'Word', 'Score', 'NewsId']
    d = {
        'Id': list(range(len(result))),
        'Word': result[0],
        'Score': result[1],
        'NewsId': result[2]
    }
    df = pd.DataFrame(d)
    df.to_csv(f'{params.apl["path2read"]}/NewNewsTagmeAnnotated.csv')
    return result

def main(NewsTable, newsstat=True):
    Text = NewsTable[params.apl["Text_Title"]].dropna()
    NewsIds = Text.index
    np.save(f'{params.apl["path2save"]}/NewsIds.npy', NewsIds)

    Text = Text.values
    processed_docs = np.asarray([news.split(',') for news in Text])

    DicPath = glob.glob(f'{params.tml["path2save"]}/*topics_TopicModelingDictionary.mm')[0]
    dictionary = gensim.corpora.Dictionary.load(DicPath)

    # LDA Model Loading
    model_name = glob.glob(f'{params.tml["path2save"]}/*.model')[0]
    GenMal = model_name.split('\\')[-1].split('_')[0]
    if GenMal == 'gensim':
        cmn.logger.info(f"Loading LDA model (Gensim) ...")
        ldaModel = gensim.models.ldamodel.LdaModel.load(model_name)
    elif GenMal == 'mallet':
        cmn.logger.info(f"Loading LDA model (Mallet) ...")
        ldaModel = gensim.models.wrappers.LdaMallet.load(model_name)
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)

    # topics = ldaModel.get_document_topics(bow_corpus)
    totalNewsTopics = []
    for news in range(len(processed_docs)):
        news_bow_corpus = dictionary.doc2bow(processed_docs[news])
        topics = tm.doc2topics(ldaModel, news_bow_corpus, threshold=params.evl['Threshold'], justOne=params.tml['JO'], binary=params.tml['Bin'])
        totalNewsTopics.append(topics)

    totalNewsTopics = np.asarray(totalNewsTopics)
    np.save(f'{params.apl["path2save"]}/NewsTopics.npy', totalNewsTopics)


