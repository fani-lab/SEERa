import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import gensim
import numpy as np
#import tagme
import os, sys
from gensim.parsing.preprocessing import STOPWORDS
#from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
import datetime
#tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"


#sys.path.extend(["../"])
from cmn import Common as cmn
import params

def drop_user(dataset, userId):



def data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=params.dal['lastRowsNumber'],
                     startDate = params.dal['start'], timeInterval=params.dal['timeInterval']):
    global DataLen
    date_time_obj = datetime.datetime.strptime(startDate, '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    newDateTime = []
    for i in range(len(dataset)):
        date = dataset[i, 2].toordinal() - startDateOrdinal
        exeededdays = date % timeInterval
        newDateTime.append((dataset[i, 2] - datetime.timedelta(exeededdays)).date())
    dataset = np.c_[dataset, np.asarray(newDateTime)]
    cmn.logger.info(f'DataPreperation: userModeling={userModeling}, timeModeling={timeModeling},'
                    f'preProcessing={preProcessing}, TagME={TagME}')

    data = pd.DataFrame({'Id': dataset[:, 0], 'Text': dataset[:, 1], 'OriginalCreationDate': dataset[:, 2],
                         'CreationDate': dataset[:, 6], 'userId': dataset[:, 3],
                         'ModificationTimeStamp': dataset[:, 4], 'Tokens': dataset[:, 5]})
    for u in data['userId'].unique():
        print('salam')
    data.to_csv(f"../output/{params.general['RunId']}/data1.csv", sep=",", encoding='utf-8', index=False)
    #data.sort_values(by=['CreationDate']) #moved to sql query

    # data.sample(frac=1)
    cmn.logger.info(f'DataPreperation: {np.abs(lastRowsNumber)} sampled from the end of dataset (sorted by creationTime)')
    data = data[-lastRowsNumber:]
    data.to_csv(f"../output/{params.general['RunId']}/data2.csv", sep=",", encoding='utf-8', index=False)

    if userModeling and timeModeling:
        dataGroupbyUsersTime = data.groupby(['userId', 'CreationDate'])['Tokens'].apply(lambda x: ','.join(x)).reset_index()
        documents = dataGroupbyUsersTime
        #print(dataGroupbyUsersTime.shape)
        #documents = dataGroupbyUsersTime['Text'].apply(lambda x: ','.join(x)).reset_index()
    elif userModeling:
        dataGroupbyUsers = data.groupby(['userId'])
        documents = dataGroupbyUsers['Tokens'].apply(lambda x: ','.join(x)).reset_index()
    elif timeModeling:
        dataGroupbyTime = data.groupby(['CreationDate'])
        documents = dataGroupbyTime['Tokens'].apply(lambda x: ','.join(x)).reset_index()
    else:
        data_text = data[['Text']]
        data_text['index'] = data_text.index
        documents = data_text
    documents.to_csv(f"../output/{params.general['RunId']}/documents_data3.csv", sep=",",
                     encoding='utf-8', index=False)
    DataLen = len(documents)
    cmn.logger.info(f'DataPreperation: Length of the dataset after applying groupby: {DataLen} \n')
    if preProcessing:
        processed_docs = documents['Text'].map(preprocess)
    elif TagME:
        processed_docs = documents['Text'].map(TAGME)
    else:
        processed_docs = []
        for tweet in documents['Tokens']:
            processed_docs.append(tweet.split(','))
    prosdocs = np.asarray(processed_docs)
    cmn.logger.info(f'DataPreparation: Processed docs shape: {prosdocs.shape}')

    return prosdocs, documents


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