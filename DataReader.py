import mysql.connector
import numpy as np
def DataReader():
    cnx = mysql.connector.connect(user='root', password='Ghsss.34436673',
                                  host='localhost',
                                  database='twitter2')
    print('Connection Created')
    cursor = cnx.cursor()
    ##cursor.execute("SELECT Text FROM tweets")
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
