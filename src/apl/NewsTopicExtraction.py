import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import glob
import pickle
import os, sys
import tagme
from tqdm import tqdm

import Params
from tml import TopicModeling as tm
from cmn import Common as cmn
from dal import DataPreparation as dp


def text2tagme(news_table, threshold=0.05):
    for i in range(len(news_table)):
        text = news_table[Params.apl["textTitle"]][i]
        annotations = tagme.annotate(text)
        result = []
        if annotations is not None:
            for keyword in annotations.get_annotations(threshold):
                result.append([keyword.entity_title, keyword.score, news_table.index[i]])
    d = {
        'Id': list(range(len(result))),
        'Word': result[0],
        'Score': result[1],
        'NewsId': result[2]
    }
    df = pd.DataFrame(d)
    df.to_csv(f'{Params.apl["path2read"]}/NewsTagmeAnnotated.csv')
    return result

def main(news_table):
    news_table = news_table[news_table[Params.apl["textTitle"]].notna()]
    news_ids = news_table.NewsId
    np.save(f'{Params.apl["path2save"]}/NewsIds_ExpandedURLs.npy', news_ids)
    if Params.dal['tagMe']:
        tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"
        for doc in tqdm(news_table.itertuples(), total=news_table.shape[0]):
            news_table.at[doc.Index, Params.apl["textTitle"]] = dp.tagme_annotator(doc.Text).split()
        processed_docs = news_table[Params.apl["textTitle"]]
    else:
        processed_docs = np.asarray([dp.preprocess(news).split() for news in news_table[Params.apl["textTitle"]]])
    dict_path = glob.glob(f'{Params.tml["path2save"]}/*TopicsDictionary.mm')[0]
    dictionary = gensim.corpora.Dictionary.load(dict_path)

    # LDA Model Loading
    if Params.tml['method'].lower() == 'gsdmm':
        model_name = glob.glob(f'{Params.tml["path2save"]}/*Topics.pkl')[0]
        with open(model_name, 'rb') as g: tm_model = pickle.load(g)
    else:
        method = Params.tml["method"].split('.')[-1]
        model_name = glob.glob(f'{Params.tml["path2save"]}/*.model')[0]
        cmn.logger.info(f'Loading {Params.tml["method"]} model ...')
        if method.lower() == 'gensim':
            tm_model = gensim.models.ldamodel.LdaModel.load(model_name)
        elif method.lower == 'mallet':
            tm_model = gensim.models.wrappers.LdaMallet.load(model_name)
            tm_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(tm_model)

    # topics = ldaModel.get_document_topics(bow_corpus)
    total_news_topics = []
    for news in range(len(processed_docs)):
        news_bow_corpus = dictionary.doc2bow(processed_docs[news])
        topics = tm.doc2topics(tm_model, news_bow_corpus, threshold=Params.evl['threshold'], just_one=Params.tml['justOne'], binary=Params.tml['binary'])
        total_news_topics.append(topics)

    np.save(f'{Params.apl["path2save"]}/NewsTopics.npy', np.asarray(total_news_topics))


