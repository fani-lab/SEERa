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


def stats(news):
    file_object = open(f"{params.apl['path2save']}/NewsStat.txt", 'a')
    #news = pd.read_csv('News.csv')
    texts = news.Text.dropna()
    titles = news.Title.dropna()
    desc = news.Description.dropna()
    print("Available texts: ", len(texts))
    file_object.write(f'Available texts: {len(texts)}\n')
    print("Available titles: ", len(titles))
    file_object.write(f'Available titles: {len(titles)}\n')
    print("Available descriptions: ", len(desc))
    file_object.write(f'Available descriptions: {len(desc)}\n')

    sumtext=0
    for i in range(len(texts)):
        sumtext += len(texts.values[i].split())
    textavg = sumtext//i
    print("Average texts length: ", textavg)
    file_object.write(f'Average texts length: {textavg} words.\n')

    sumtitle=0
    for i in range(len(titles)):
        sumtitle += len(titles.values[i].split())
    titleavg = sumtitle//i
    print("Average titles length: ", titleavg)
    file_object.write(f'Average titles length: {titleavg} words.\n')

    sumdesc=0
    for i in range(len(desc)):
        sumdesc += len(desc.values[i].split())
    descavg = sumdesc//i
    print("Average descriptions length: ", descavg)
    file_object.write(f'Average descriptions length: {descavg} words.\n')
    file_object.close()


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
    #cmn.logger.info("\nNewsTopicExtraction.py:\n")
    try:
        Text = NewsTable[params.apl["Text_Title"]].dropna()
    except:
        print("Text_Title in params.py must be Text or Title.")
    NewsIds = Text.index
    np.save(f'{params.apl["path2save"]}/NewsIds.npy', NewsIds)

    Text = Text.values
    processed_docs = []
    for news in Text:
        processed_docs.append(news.split(','))
    processed_docs = np.asarray(processed_docs)
    print(glob.glob(f'{params.tml["path2save"]}/*topics_TopicModelingDictionary.mm'))
    print(f'{params.tml["path2save"]}/*topics_TopicModelingDictionary.mm')
    DicPath = glob.glob(f'{params.tml["path2save"]}/*topics_TopicModelingDictionary.mm')[0]
    dictionary = gensim.corpora.Dictionary.load(DicPath)
    ## bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # LDA Model Loading
    model_name = glob.glob(f'{params.tml["path2save"]}/*.model')[0]
    print('model name:', model_name)
    #cmn.logger.critical("model "+model_name + " is loaded.")
    GenMal = model_name.split('\\')[-1].split('_')[0]
    if GenMal == 'gensim':
        ldaModel = gensim.models.ldamodel.LdaModel.load(model_name)
        print('Lda Model Loaded (Gensim)')
    elif GenMal == 'mallet':
        ldaModel = gensim.models.wrappers.LdaMallet.load(model_name)
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        print('Lda Model Loaded (Mallet)')
    else:
        print('Wrong Library!')

    # topics = ldaModel.get_document_topics(bow_corpus)
    totalNewsTopics = []
    for news in range(len(processed_docs)):
        news_bow_corpus = dictionary.doc2bow(processed_docs[news])
        topics = tm.doc2topics(ldaModel, news_bow_corpus, threshold=params.evl['Threshold'], justOne=params.tml['JO'],
                               binary=params.tml['Bin'])
        totalNewsTopics.append(topics)

    #cmn.logger.critical("Topics are extracted for news dataset based on the tweets extracted topics.\n")
    totalNewsTopics = np.asarray(totalNewsTopics)
    #cmn.save2excel(totalNewsTopics, 'evl/totalNewsTopics')
    np.save(f'{params.apl["path2save"]}/NewsTopics.npy', totalNewsTopics)
    if newsstat:
        stats(NewsTable)

