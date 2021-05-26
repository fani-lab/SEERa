import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import gensim
import numpy as np
import tagme
import os, sys
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"


sys.path.extend(["../"])
from cmn import Common as cmn

def data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=0):
    global DataLen
    cmn.logger.info(f'DataPreperation: userModeling={userModeling}, timeModeling={timeModeling}, preProcessing={preProcessing}, TagME={TagME}')
    data = pd.DataFrame({'Id': dataset[:, 0], 'Text': dataset[:, 1], 'CreationDate': dataset[:, 2], 'userId': dataset[:, 3], 'ModificationTimeStamp': dataset[:, 4]})
    #data.sort_values(by=['CreationDate']) #moved to sql query

    #print(data['CreationDate'])
    # data.sample(frac=1)
    cmn.logger.info(f'DataPreperation: {np.abs(lastRowsNumber)} sampled from the end of dataset (sorted by creationTime)')
    data = data[-lastRowsNumber:]
    if userModeling and timeModeling:
        dataGroupbyUsersTime = data.groupby(['userId', 'CreationDate'])
        documents = dataGroupbyUsersTime['Text'].apply(lambda x: ','.join(x)).reset_index()
    elif userModeling:
        dataGroupbyUsers = data.groupby(['userId'])
        documents = dataGroupbyUsers['Text'].apply(lambda x: ','.join(x)).reset_index()
    elif timeModeling:
        dataGroupbyTime = data.groupby(['CreationDate'])
        documents = dataGroupbyTime['Text'].apply(lambda x: ','.join(x)).reset_index()
    else:
        data_text = data[['Text']]
        data_text['index'] = data_text.index
        documents = data_text
    DataLen = len(documents)
    cmn.logger.info(f'DataPreperation: Length of the dataset after applying groupby: {DataLen} \n')
    if preProcessing:
        processed_docs = documents['Text'].map(preprocess)
    elif TagME:
        processed_docs = documents['Text'].map(TAGME)
    else:
        processed_docs = []
        for tweet in documents['Text']:
            processed_docs.append(tweet.split(','))
    return processed_docs, documents


def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    global DataLen
    result = []
    stemmer = SnowballStemmer('english')
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token, stemmer))
    return result


def TAGME(text, threshold=0.05):
    global DataLen
    annotations = tagme.annotate(text)
    result = []
    if annotations is not None:
        for keyword in annotations.get_annotations(threshold):
            result.append(keyword.entity_title)
    return result

## test
# dataset = dr.load_tweets(Tagme=True, start='2011-01-01', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'])
# data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=-1000)