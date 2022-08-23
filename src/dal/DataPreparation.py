import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
import gensim
import numpy as np
from tqdm import tqdm
import tagme
import os, sys, re
from gensim.parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
global TagmeCounter
global DataLen
import datetime

from cmn import Common as cmn
import Params

def data_preparation(posts, userModeling, timeModeling, TagME, startDate, timeInterval, stopwords=['www', 'RT', 'com', 'http']):
    date_time_obj = datetime.datetime.strptime(startDate, '%Y-%m-%d').date()
    startDateOrdinal = date_time_obj.toordinal()
    posts_temp = []
    for post in tqdm(posts.itertuples(), total=posts.shape[0]):
        date = post.CreationDate.toordinal() - startDateOrdinal
        exeededdays = date % timeInterval
        post = post._replace(CreationDate=(post.CreationDate - datetime.timedelta(exeededdays)).date())
        post = post._replace(Text=preprocess(post.Text))
        posts_temp.append(post)
    posts = pd.DataFrame(posts_temp)
    n_users = len(posts['UserId'].unique())
    n_timeintervals = len(posts['CreationDate'].unique())
    #assert (not userModeling or timeModeling) #if usermodeling then timemodeling
    if userModeling and timeModeling: documents = posts.groupby(['UserId', 'CreationDate'])['Text'].apply(lambda x: ' '.join(x)).reset_index()
    elif userModeling:
        documents = posts.groupby(['UserId'])['Text'].apply(lambda x: ' '.join(x)).reset_index()
        dates = posts['CreationDate']
        dates[dates != Params.dal['end']] = datetime.datetime.strptime(startDate, '%Y-%m-%d')
        documents.insert(1, 'CreationDate', dates)
    elif timeModeling:
        documents = posts.groupby(['CreationDate'])['Text'].apply(lambda x: ' '.join(x)).reset_index()
        documents.insert(0, 'UserId', np.ones(len(documents)))
    else:
        documents = posts[['UserId', 'Text', 'CreationDate']]

    if TagME: #cannot be sooner because TagMe needs context, the more the better
        import tagme
        tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"
        for doc in tqdm(documents.itertuples(), total=documents.shape[0]):
            documents.at[doc.Index, 'Text'] = tagme_annotator(doc.Text)
    documents.to_csv(f"../output/{Params.general['baseline']}/Documents.csv", encoding='utf-8', index=False)
    prosdocs = np.asarray(documents['Text'].str.split())
    np.savez_compressed(f"../output/{Params.general['baseline']}/Prosdocs.npz", a=prosdocs)

    return prosdocs, documents, n_users, n_timeintervals

def lemmatize_stemming(text, stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    # stemmer = SnowballStemmer('english')
    text = re.sub('RT @\w +: ', ' ', text)
    text = re.sub('(@[A-Za-z0–9]+) | ([0-9A-Za-z \t]) | (\w+:\ / \ / \S+)', ' ', text)
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = text.lower()
    result = [token for token in gensim.utils.simple_preprocess(text) if token not in STOPWORDS and len(token) > 2]

    return ' '.join(result)

def tagme_annotator(text, threshold=0.05):
    annotations = tagme.annotate(text)
    result = []
    if annotations is not None:
        for keyword in annotations.get_annotations(threshold):
            result.append(keyword.entity_title)
    return ' '.join(result)

## test
# dataset = dr.load_tweets(Tagme=True, start='2011-01-01', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'])
# data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=-1000)