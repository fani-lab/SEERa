import sys, re
import pandas as pd
import datetime

def NewsGoldenStandard2Reader():
    return pd.read_csv(r'../../data/NewsGoldenStandard.csv', sep=';', encoding='utf-8')

def load_tweets(path, startDate, endDate, stopwords=['www', 'RT', 'com', 'http']):
    start = datetime.datetime.strptime(startDate, '%Y-%m-%d')
    end = datetime.datetime.strptime(endDate, '%Y-%m-%d')

    tweets = pd.read_csv(path, encoding='utf-8', parse_dates=['CreationTimestamp'])
    tweets.rename(columns={'Id': 'TweetId', 'CreationTimestamp': 'CreationDate'}, inplace=True)
    tweets = tweets[tweets.TweetId != -1]  # remove rows with tweet ids -1 value
    tweets = tweets[tweets.UserId != -1]  # remove rows with user ids with -1 value
    tweets = tweets.loc[(tweets['CreationDate'].dt.date >= start.date()) & (tweets['CreationDate'].dt.date <= end.date())]
    tweets['ModificationTimestamp'] = pd.to_datetime(tweets['ModificationTimestamp'])
    tweets['CreationDate'] = pd.to_datetime(tweets['CreationDate'])  # declare that creationdate and modification timestamp is a datetime value

    remove_rt = lambda x: re.sub('RT @\w +: ', ' ', x)
    rt = lambda x: re.sub('(@[A-Za-z0–9]+) | ([0-9A-Za-z \t]) | (\w+:\ / \ / \S+)', ' ', x)
    tweets['Tokens'] = tweets['Text'].map(remove_rt).map(rt).str.lower()

    return tweets

## test
#path= ['../../data/tweets_N20000.csv', '../../data/tagmeannotation_N80000.csv']
#a = load_tweets(path=path, startDate='2000-10-06', endDate='2021-01-01', stopwords=['www', 'RT', 'com', 'http'], tagme_threshold=0.07)

## converting tweets in mysql database to csv file
# pip install mysqlclient
# pip install sqlalchemy
# import pandas as pd
# from sqlalchemy import create_engine
# engine = create_engine("mysql+mysqldb://userid:password@localhost/twitter")
#
# sql="select * from tweets where CreationTimestamp between '2010-12-01' and '2010-12-05'"
# my_data = pd.read_sql(sql, engine)
# my_data.to_csv('Twitter.csv', index=False)































