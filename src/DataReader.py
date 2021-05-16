import mysql.connector
import numpy as np


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

def DataReader(Tagme = True):
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673',
                                  host='localhost',
                                  database='twitter3')
    print('Connection Created')
    cursor = cnx.cursor()
    if Tagme:
        sqlScript = '''
                (SELECT TweetId,GROUP_CONCAT('', Word) as Tokens,Date(CreationTimestamp) as CreationDate,UserId,ModificationTimestamp
                FROM
                (SELECT TweetId,CreationTimestamp,UserId,ModificationTimestamp,Word
                FROM
                 tweets inner join TagMeAnnotations on tweets.Id = TagMeAnnotations.TweetId where
                 tweets.CreationTimeStamp between '2010-11-08' and '2010-11-18' and Tweets.Id != -1 and Tweets.UserId != -1 and TagMeAnnotations.Score > 0.07 and TagMeAnnotations.Word not in ('www', 'RT', 'com', 'http')) AS T
                GROUP BY TweetId);
                '''
        cursor.execute(sqlScript)
    else:
        cursor.execute("SELECT * FROM tweets")
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    print(table[0])
    print('Connection Closed')
    return table

def GoldenStandard2Reader():
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673',
                                  host='localhost',
                                  database='twitter3')
    print('Connection Created')
    cursor = cnx.cursor()
    sqlScript = '''
            (SELECT * FROM GoldenStandard2)
            '''
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    print('Connection Closed')
    return table
