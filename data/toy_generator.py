import pandas as pd
import numpy as np


te = pd.read_csv('main_tables_csv/TweetEntities.csv', delimiter=';')
t = pd.read_csv('toy/Tweets.csv')
ids = t.Id.values
s = []
for index, row in te.iterrows():
    if row['TweetId'] in ids:
        s.append(row)
