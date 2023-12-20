import datetime

import pandas as pd
from cmn import Common as cmn
import params
import re


def data_preparation(dataset):
    # Remove columns we don't need / rows with Nan
    cols_to_drop = ['ModificationTimestamp']
    dataset.dropna(inplace=True)
    dataset.drop(cols_to_drop, axis=1, inplace=True)

    # Reassign the TweetIDs and UserIDs (makes it easier, because here the IDs didn't start at 0)
    posts = dataset['TweetId'].unique()
    new_ids = list(range(len(dataset['TweetId'].unique())))
    mapping = dict(zip(posts, new_ids))
    dataset['TweetId'] = dataset['TweetId'].map(mapping)

    posts = dataset['UserId'].unique()
    new_ids = list(range(len(dataset['UserId'].unique())))
    mapping = dict(zip(posts, new_ids))
    dataset['UserId'] = dataset['UserId'].map(mapping)

    dataset = dataset.sort_values(by="CreationDate")
    # Adding TimeStamp to the dataset
    date_time_obj = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    timeStamps = []
    for index, row in dataset.iterrows():
        dayDiff = row['CreationDate'].toordinal() - startDateOrdinal
        timeStamps.append((dayDiff // params.dal['timeInterval']))
    dataset['TimeStamp'] = timeStamps

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

    documents.to_csv(f"../output/{params.general['baseline']}/documents.csv", encoding='utf-8', index=False, header=True)
    cmn.logger.info(f'DataPreparation: Documents shape: {documents.shape}')
    return documents

def preprocess_tweets(text):
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt')
    nltk.download('stopwords')
    preprocessed_text = text.apply(lambda x: x.lower())
    # Remove mentions (@username)
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'@\w+', '', x))
    # Remove special characters and punctuation
    preprocessed_text = preprocessed_text.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    # Tokenization
    preprocessed_text = preprocessed_text.apply(word_tokenize)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    gist_file = open(params.dal['stopwordPath'], "r")
    try:
        content = gist_file.read()
        stopwords2 = content.split(",")
    finally:
        gist_file.close()
    preprocessed_text = preprocessed_text.apply(lambda tokens: [token for token in tokens if token not in stop_words and token not in stopwords2])

    # Remove one-letter and two-letter tokens, and tokens longer than 10 characters
    preprocessed_text = preprocessed_text.apply(lambda tokens: [token for token in tokens if (len(token) > 2 and len(token) <= 10)])

    return pd.Series(preprocessed_text)
