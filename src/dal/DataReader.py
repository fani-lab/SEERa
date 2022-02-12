import mysql.connector
import numpy as np
import sys

# MySQL passwords:
# Surface: Ghsss.34436673
# Y510: soroush56673sor7
import pandas as pd

sys.path.extend(["../"])
from cmn import Common as cmn
import params

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
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
    cmn.logger.info('DataReader: Connection created')
    cursor = cnx.cursor()
    if Tagme:
        sqlScript = f'''
                (SELECT TweetId, GROUP_CONCAT('', Word) as Tokens, Date(CreationTimestamp) as CreationDate, UserId, ModificationTimestamp
                FROM
                (SELECT TweetId, CreationTimestamp, UserId, ModificationTimestamp, Word
                FROM
                 Tweets INNER JOIN TagMeAnnotations ON Tweets.Id = TagMeAnnotations.TweetId 
                 where Tweets.CreationTimeStamp BETWEEN '{start} 00:00:00' AND '{end} 23:59:59' AND 
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
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
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


##########################################################################################################
#########################################################################################################
def GoldenStandard2ReaderCSV():
    table = pd.read_csv(r'../CSV/goldenstandard.csv', sep=';',
                        encoding='utf-8') #loads all from GoldenStandard CSV into dataframe
    return table


def load_tweetsCVS(startDate, endDate, stopwords=['www', 'RT', 'com', 'http']):

    # reading csv file and creating dataframe with name tweets, change as necessary
    tweets = pd.read_csv(r'../CSV/stweets.csv', sep=';', encoding='utf-8')
    # changing name of Id to TweetId to allow merge with tagme dataframe
    tweets.rename(columns={'Id': 'TweetId'}, inplace=True)
    tweets = tweets.drop(columns=["Text"])  # drop unwanted columns named Text
    tweets = tweets[tweets.TweetId != -1]  # remove rows with tweet ids -1 value
    tweets = tweets[tweets.UserId != -1]  # remove rows with user ids with -1 value

    tagme = pd.read_csv(r'../CSV/tagmeannotations.csv', sep=';', encoding='utf-8')  # creating tagme dataframe from csv
    tagme = tagme[tagme.Score > 0.07]  # drop scores below .07
    tagme = tagme[~tagme['Word'].isin(stopwords)]  # drop rows with stopwords
    tagme = tagme.drop(columns=["StartIndex", "EndIndex", "ConceptId", "IsDeleted", "Id",
                                "Score"])  # drop unwanted columns
    tagme.Word = tagme.Word.astype('string')  # declare type of word as string
    tagme = tagme.groupby('TweetId').agg(
        lambda word: word.tolist())  # take all words with same TweetId and put them in a list

    tweets = pd.merge(tweets, tagme, on='TweetId')  # merge dataframes
    tweets.rename(columns={"CreationTimestamp": "CreationDate", "Word": "Tokens"},
                  inplace=True)  # rename columns to match sql quarry

    tweets['ModificationTimestamp'] = pd.to_datetime(tweets['ModificationTimestamp'])
    tweets['CreationDate'] = pd.to_datetime(tweets['CreationDate']) # declare that creationdate and modification timestamp is a datetime value

    # locate only the tweets we want to actually grab and save to tweets dataframe from the entered start and end date
    tweets = tweets.loc[(tweets['CreationDate'] >= startDate)
                        & (tweets['CreationDate'] < endDate)]

    print(tweets.to_string()) # for viewing table during testing
    return tweets
#########################################################################################################
#########################################################################################################






































