import mysql.connector
import numpy as np
import sys

sys.path.extend(["../"])
from cmn import Common as cmn

'''
create table GoldenStandard2 as
(Select  T.Id, userId, NewsId, T.URL, T.CreationTimestamp, T.tweetId from
(select goldenstandard.UserId, goldenstandard.Id as Id, goldenstandard.Url, NewsId, goldenstandard.CreationTimestamp, tweets.Id as tweetId from 
	goldenstandard inner join 
    tweets on 
    (tweets.UserId = goldenstandard.UserId and tweets.CreationTimestamp = goldenstandard.CreationTimestamp)) as T
	inner join tweetentities on
    T.URL = tweetentities.ExpandedURL and T.tweetId = tweetentities.TweetId)
'''

def load_tweets(Tagme=True, start='2010-11-08', end='2010-11-18', stopwords=['www', 'RT', 'com', 'http']):
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673', host='localhost', database='twitter3')
    cmn.logger.info('DataReader: Connection created')
    cursor = cnx.cursor()
    if Tagme:
        sqlScript = f'''
                (SELECT TweetId, GROUP_CONCAT('', Word) as Tokens, Date(CreationTimestamp) as CreationDate, UserId, ModificationTimestamp
                FROM
                (SELECT TweetId, CreationTimestamp, UserId, ModificationTimestamp, Word
                FROM
                 Tweets INNER JOIN TagMeAnnotations ON Tweets.Id = TagMeAnnotations.TweetId 
                 where Tweets.CreationTimeStamp BETWEEN '{start}' AND '{end}' AND 
                       Tweets.Id != -1 and Tweets.UserId != -1 AND 
                       TagMeAnnotations.Score > 0.07 AND 
                       TagMeAnnotations.Word NOT IN ("{'","'.join(stopwords)}")) AS T
                GROUP BY TweetId
                ORDER BY CreationDate);
                '''
        cursor.execute(sqlScript)
    else:
        cursor.execute("SELECT * FROM Tweets")
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    cmn.logger.info(f'DataReader: {table.shape[0]} rows returned')
    cmn.logger.info('DataReader: Connection closed')
    return table

def GoldenStandard2Reader():
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673', host='localhost', database='twitter3')
    cmn.logger.info('DataReader: Connection created')
    cursor = cnx.cursor()
    sqlScript = '''
            (SELECT * FROM GoldenStandard2)
            '''
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    cmn.logger.info('DataReader: Connection closed')
    return table

## test
#load_tweets(Tagme=True, start='2015-11-01', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'])
