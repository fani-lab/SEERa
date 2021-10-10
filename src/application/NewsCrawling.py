import sys

sys.path.extend(["../"])
import params
import numpy as np
import mysql.connector
from newspaper import Article

def createTable(cursor):

    cursor.execute("DROP TABLE IF EXISTS News_c")
    sql = '''CREATE TABLE News_c(
       Id bigint NOT NULL,
       TweetId bigint,
       Text text,
       Title VARCHAR(255),
       Url text
    )'''
    cursor.execute(sql)
def insert(cnx, cursor, id, tid, text, title, url):
    sql = f'''INSERT INTO News_C(
             Id, TweetId, Text, Title, Url)
             VALUES ({id}, {tid}, '{text}', '{title}', '{url}')'''
    # print(sql)
    cursor.execute(sql)
    try:
        cnx.commit()
    except:
        cnx.rollback()
    print(f"Value added {id}, {tid}, '{text}', '{title}', '{url}'")
cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
cursor = cnx.cursor()
createTable(cursor)
sqlScript = 'SELECT distinct Id, ExpandedUrl FROM twitter3.tweetentities where ExpandedUrl is not null'
cursor.execute(sqlScript)
result = cursor.fetchall()
result = np.asarray(result)
Ids = result[:, 0]
URLs = result[:, 1]
newsArticles = []
newsTitles = []
TweetIds = []
for count, url in enumerate(URLs):
    try:
        # publishDate = article.publish_date
        # print('publish date:', publishDate)
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        text = text.replace("'", "")
        title = article.title
        title = title.replace("'", "")
        newsArticles.append(text)
        newsTitles.append(title)
        TweetIds.append(Ids[count])
        insert(cnx, cursor, count+1, Ids[count], text, title, url)

    except:
        continue

cnx.close()
