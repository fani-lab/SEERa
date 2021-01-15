import mysql.connector
import numpy as np


def DataReader(Tagme = True):
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673',
                                  host='localhost',
                                  database='twitter3')
    print('Connection Created')
    cursor = cnx.cursor()
    if Tagme:
        sqlScript = '''
        (SELECT TweetId,GROUP_CONCAT(' ', Word) as Tokens,CreationTimestamp,UserId,ModificationTimestamp
        FROM
        (SELECT TweetId,CreationTimestamp,UserId,ModificationTimestamp,Word FROM
         tweets inner join tagmeannotations on tweets.Id = tagmeannotations.TweetId) AS T
        GROUP BY TweetId);
        '''
        cursor.execute(sqlScript)
    else:
        cursor.execute("SELECT * FROM tweets")
    result = cursor.fetchall()
    table = np.asarray(result)
    ##TweetID = Table[:,0]
    ##Tweets =  Table[:,1]
    ##CreationTime = Table[:,2]
    ##UserIDs = Table[:,3]
    ##ModificationTime = Tweets[:,4]
    cnx.close()
    print('Connection Closed')
    return table
