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
    posts.dropna(subset=['Text'], inplace=True)
    posts = posts.loc[(posts['CreationDate'].dt.date >= start.date()) & (posts['CreationDate'].dt.date <= end.date())]
    posts['CreationDate'] = pd.to_datetime(posts['CreationDate'])  # declare that creationdate and modification timestamp is a datetime value

    return posts





























