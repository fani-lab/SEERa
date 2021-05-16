import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import gensim
import numpy as np
from src.DataReader import DataReader
import tagme
import os
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"


def data_preparation(userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, logger=None):
    global DataLen
    print('data_preparation')
    logger.critical("userModeling="+str(userModeling)+" timeModeling="+str(timeModeling)+" preProcessing="+str(preProcessing)+" TagME="+str(TagME)+'\n')
    dataset = DataReader(Tagme=True)
    data = pd.DataFrame(
             {'Id': dataset[:, 0], 'Text': dataset[:, 1], 'CreationDate': dataset[:, 2], 'userId': dataset[:, 3],
              'ModificationTimeStamp': dataset[:, 4]})
    data.sort_values(by=['CreationDate'])
    print(len(data))
    logger.critical("len(data):"+str(len(data))+'\n')
    print(data['CreationDate'])
    # data.sample(frac=1)
    lastRowsNumber = -10000
    logger.critical("How many rows are chosen from the end of dataset (sorted by creationTime):" + str(np.abs(lastRowsNumber))+'\n')
    data = data[lastRowsNumber:]
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
    logger.critical("Length of the dataset after applying groupby:" + str(DataLen)+'\n')
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
        if token not in STOPWORDS and len(token) > 2 and token != 'http' and token != 'RT':
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
