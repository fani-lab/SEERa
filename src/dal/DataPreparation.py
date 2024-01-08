import datetime

import pandas as pd
from cmn import Common as cmn
import params
import re

def reassign_id(table, column):
    posts = table[column].unique()
    new_ids = list(range(len(table[column].unique())))
    mapping = dict(zip(posts, new_ids))
    table[column] = table[column].map(mapping)
    return table

def date2timestamp(table,date_col):
    date_time_obj = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    timeStamps = []
    for index, row in table.iterrows():
        dayDiff = row[date_col].toordinal() - startDateOrdinal
        timeStamps.append((dayDiff // params.dal['timeInterval']))
    table['TimeStamp'] = timeStamps
    return table

def preprocess_tweets(text):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    preprocessed_text = text.apply(lambda x: x.lower())
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'@\w+', '', x))
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    preprocessed_text = preprocessed_text.apply(word_tokenize)
    stop_words = set(stopwords.words('english'))
    gist_file = open(params.dal['stopwordPath'], "r")
    try:
        content = gist_file.read()
        stopwords2 = content.split(",")
    finally:
        gist_file.close()
    preprocessed_text = preprocessed_text.apply(lambda tokens: [token for token in tokens if token not in stop_words and token not in stopwords2 and 2 < len(token) <= 10])
    return pd.Series(preprocessed_text)


def data_preparation(dataset):
    cols_to_drop = ['ModificationTimestamp']
    dataset.dropna(inplace=True)
    dataset.drop(cols_to_drop, axis=1, inplace=True)
    dataset = dataset.sort_values(by="CreationDate")
    dataset = date2timestamp(dataset, 'CreationDate')
    cmn.logger.info(f'DataPreperation: userModeling={params.dal["userModeling"]}, timeModeling={params.dal["timeModeling"]}, preProcessing={params.dal["preProcessing"]}, TagME={params.dal["tagMe"]}')
    if params.dal['userModeling'] and params.dal['timeModeling']:
        documents = dataset.groupby(['UserId', 'TimeStamp']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    elif params.dal['userModeling']:
        documents = dataset.groupby(['UserId']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    elif params.dal['timeModeling']:
        documents = dataset.groupby(['TimeStamp']).agg({'Text': lambda x: ' '.join(x), 'Extracted_Links': 'sum'}).reset_index()
    else:
        documents = dataset
    if params.dal['preProcessing']: documents['Tokens'] = preprocess_tweets(documents['Text'])
    else: documents['Tokens'] = documents['Text'].str.split()
    documents.to_csv(f"../output/{params.general['baseline']}/documents.csv", encoding='utf-8', index=False,header=True)
    cmn.logger.info(f'DataPreparation: Documents shape: {documents.shape}')
    return documents