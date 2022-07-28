import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import gensim
import numpy as np

import os, sys
from gensim.parsing.preprocessing import STOPWORDS
#from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
import datetime

from cmn import Common as cmn
import params

def data_preparation(dataset, userModeling, timeModeling, preProcessing, TagME, startDate, timeInterval):

    date_time_obj = datetime.datetime.strptime(startDate, '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    newDateTime = []
    for i in range(len(dataset)):
        date = dataset[i, 2].toordinal() - startDateOrdinal
        exeededdays = date % timeInterval
        newDateTime.append((dataset[i, 2] - datetime.timedelta(exeededdays)).date())
    dataset = np.c_[dataset, np.asarray(newDateTime)]
    cmn.logger.info(f'DataPreperation: userModeling={userModeling}, timeModeling={timeModeling}, preProcessing={preProcessing}, TagME={TagME}')

    data = pd.DataFrame({'Id': dataset[:, 0], 'Text': dataset[:, 1], 'OriginalCreationDate': dataset[:, 2],
                         'CreationDate': dataset[:, 6], 'userId': dataset[:, 3],
                         'ModificationTimeStamp': dataset[:, 4], 'Tokens': dataset[:, 5]})

    if userModeling and timeModeling: documents = data.groupby(['userId', 'CreationDate'])['Tokens'].apply(lambda x: ' '.join(x)).reset_index()
    elif userModeling: documents = data.groupby(['userId'])['Tokens'].apply(lambda x: ' '.join(x)).reset_index()
    elif timeModeling: documents = data.groupby(['CreationDate'])['Tokens'].apply(lambda x: ' '.join(x)).reset_index()
    else:
        data_text = data[['Text']]
        data_text['index'] = data_text.index
        documents = data_text

    documents.to_csv(f"../output/{params.general['runId']}/documents.csv", encoding='utf-8', index=False)
    cmn.logger.info(f'DataPreperation: Length of the dataset after applying groupby: {len(documents)} \n')

    if preProcessing: processed_docs = documents['Text'].map(preprocess)
    elif TagME:
        import tagme
        tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"
        processed_docs = documents['Text'].map(TAGME)
    else: processed_docs = documents['Tokens'].str.split()
    prosdocs = np.asarray(processed_docs)
    np.savez_compressed(f"../output/{params.general['runId']}/prosdocs.npz", a=prosdocs)
    cmn.logger.info(f'DataPreparation: Processed docs shape: {prosdocs.shape}')

    return prosdocs, documents

def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    stemmer = SnowballStemmer('english')
    result = [lemmatize_stemming(token, stemmer) for token in gensim.utils.simple_preprocess(text) if token not in STOPWORDS and len(token) > 2]
    return result

def TAGME(text, threshold=0.05):
    annotations = tagme.annotate(text)
    result = []
    if annotations is not None:
        for keyword in annotations.get_annotations(threshold):
            result.append(keyword.entity_title)
    return result

## test
# dataset = dr.load_tweets(Tagme=True, start='2011-01-01', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'])
# data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=-1000)