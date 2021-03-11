import pandas as pd
import gensim
import numpy as np
from DataReader import DataReader
import tagme
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"


def data_preparation(userModeling=True, timeModeling=True,  preProcessing=False, TagME=True):
    global DataLen
    print('data_preparation')
    dataset = DataReader(Tagme=True)
    creation_dates = []
    creation_times = []
    for i in dataset[:, 2]:
        creation_dates.append(i.date())
        creation_times.append(i.time())
    creation_dates = np.asarray(creation_dates)
    creation_times = np.asarray(creation_times)
    # data = pd.DataFrame(
    #     {'Id': dataset[:, 0], 'Text': dataset[:, 1], 'CreationTimeStamp': dataset[:, 2], 'userId': dataset[:, 3],
    #      'ModificationTimeStamp': dataset[:, 4]})
    data = pd.DataFrame(
             {'Id': dataset[:, 0], 'Text': dataset[:, 1], 'CreationDate': creation_dates, 'CreationTimes': creation_times, 'userId': dataset[:, 3],
              'ModificationTimeStamp': dataset[:, 4]})
    # data = data.sample(frac=1)
    data = data[:1000]
    if userModeling and timeModeling:
        dataGroupbyUsers = data.groupby(['userId', 'CreationDate'])
        documents = dataGroupbyUsers['Text'].apply(lambda x: ','.join(x)).reset_index()
    elif userModeling:
        dataGroupbyUsers = data.groupby(['userId'])
        documents = dataGroupbyUsers['Text'].apply(lambda x: ','.join(x)).reset_index()
    elif timeModeling:
        dataGroupbyUsers = data.groupby(['CreationDate'])
        documents = dataGroupbyUsers['Text'].apply(lambda x: ','.join(x)).reset_index()
    else:
        data_text = data[['Text']]
        data_text['index'] = data_text.index
        documents = data_text
    DataLen = len(documents)
    if preProcessing:
        processed_docs = documents['Text'].map(preprocess)
    elif TagME:
        # This part exists just for internet low quality
        # newdocument = ''
        # for i in documents['Text']:
        #     newdocument += i
        #     newdocument += ' Aaron Burr '
        # processed_docs = TAGME(newdocument)
        # processed_docs_temp = []
        # processed_docs_temp.append([])
        # for i in processed_docs:
        #     if i != 'Aaron Burr':
        #         processed_docs_temp[-1].append(i)
        #     else:
        #         print('I saw Aaron Burr!')
        #         processed_docs_temp.append([])
        # processed_docs = processed_docs_temp
        processed_docs = documents['Text'].map(TAGME)
    else:
        processed_docs = []
        for tweet in documents['Text']:
            processed_docs.append(tweet.split(','))
        # for tweet in documents['Text']:
        #     a = tweet.split()
        #     for i in range(len(a)):
        #         if not a[i][-1].isalnum():
        #             a[i] = a[i][:-1]
        #     processed_docs.append(list(set(a)))
    return processed_docs, documents


def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


PreprocessCounter = 0


def preprocess(text):
    global DataLen
    global PreprocessCounter
    if PreprocessCounter % 100 == 0:
        print(PreprocessCounter, '/', DataLen)
    PreprocessCounter += 1
    result = []
    stemmer = SnowballStemmer('english')
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 2 and token != 'http':
            result.append(lemmatize_stemming(token, stemmer))
    return result


TagmeCounter = 0


def TAGME(text, threshold=0.05):
    global DataLen
    global TagmeCounter
    if TagmeCounter % 20 == 0:
        print(TagmeCounter, '/', DataLen)
    TagmeCounter += 1
    annotations = tagme.annotate(text)
    result = []
    if annotations is not None:
        for keyword in annotations.get_annotations(threshold):
            result.append(keyword.entity_title)
    return result
