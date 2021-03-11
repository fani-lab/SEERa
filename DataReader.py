import mysql.connector
import numpy as np


def DataReader(Tagme = True):
    cnx = mysql.connector.connect(user='root', password='soroush56673sor7',
                                  host='localhost',
                                  database='CommunityPrediction')
    print('Connection Created')
    cursor = cnx.cursor()
    if Tagme:
        sqlScript = '''
        (SELECT TweetId,GROUP_CONCAT('', Word) as Tokens,CreationTimestamp,UserId,ModificationTimestamp
        FROM
        (SELECT TweetId,CreationTimestamp,UserId,ModificationTimestamp,Word FROM
         Tweets inner join TagMeAnnotations on Tweets.Id = TagMeAnnotations.TweetId Where Tweets.UserId != -1) AS T
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
