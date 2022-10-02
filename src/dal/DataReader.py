import sys
import pandas as pd
import datetime

import Params

def load_posts(path, start_date, end_date):
    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    end = end - datetime.timedelta(days=Params.dal["timeInterval"])

    posts = pd.read_csv(path, encoding='utf-8', parse_dates=['CreationTimestamp'])
    posts.rename(columns={'Id': 'TweetId', 'CreationTimestamp': 'CreationDate'}, inplace=True)
    posts = posts[posts.TweetId != -1]  # remove rows with tweet ids -1 value
    posts = posts[posts.UserId != -1]  # remove rows with user ids with -1 value
    posts = posts.loc[(posts['CreationDate'].dt.date >= start.date()) & (posts['CreationDate'].dt.date <= end.date())]
    # posts['ModificationTimestamp'] = pd.to_datetime(tweets['ModificationTimestamp'])
    posts['CreationDate'] = pd.to_datetime(posts['CreationDate'])  # declare that creationdate and modification timestamp is a datetime value

    return posts

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

# sql = "SELECT tweetentities.* FROM tweetentities INNER JOIN tweets on tweets.Id = tweetentities.TweetId and CreationTimestamp between '2010-12-01' and '2010-12-05' and EntityTypeCode = 2;"
# my_data = pd.read_sql(sql, engine)
# my_data.to_csv('TweetEntities.csv', index=False)































