import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import glob
import os
import params
from cmn import Common as cmn
def DistinctUsersandMinMaxDate():
    cnx = mysql.connector.connect(user='root', password='soroush56673sor7',
                                      host='localhost',
                                      database='CommunityPrediction')
    print('Connection Created')
    cursor = cnx.cursor()
    sqlScript = '''SELECT min(CreationTimestamp), max(CreationTimestamp) FROM GoldenStandard2'''
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    minCreationtime, maxCreationtime = result[0][0], result[0][1]
    sqlScript = '''SELECT Distinct UserId from GoldenStandard'''
    cursor.execute(sqlScript)
    DistinctUsers = cursor.fetchall()
    cnx.close()
    print('Connection Closed')
    return minCreationtime, maxCreationtime, DistinctUsers


def TextExtractor():
    cnx = mysql.connector.connect(user='root', password='soroush56673sor7',
                                  host='localhost',
                                  database='CommunityPrediction')
    cursor = cnx.cursor()
    sqlScript = '''
    SELECT NewsId,GROUP_CONCAT('', Word)
                FROM `NewsTagMeAnnotations`
                group by NewsId 
                '''
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    cnx.close()
    return result


def UserMentions(User, *args, minDate):
    cnx = mysql.connector.connect(user='root', password='soroush56673sor7',
                                  host='localhost',
                                  database='CommunityPrediction')
    cursor = cnx.cursor()
    if len(args) == 0:
        sqlScript = '''SELECT * FROM GoldenStandard where UserId = ''' + str(User)
    elif len(args) == 1:
        date = minDate + pd._libs.tslibs.timestamps.Timedelta(days=args[0])
        print(date.date())
        sqlScript = '''Select * FROM GoldenStandard where UserId = ''' + str(User) + ''' and date(Creationtimestamp) = ''' + "'" + str(date.date()) + "'"
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    table = np.asarray(result)
    cnx.close()
    return table

def DateMentions(Date, minDate):
    cnx = mysql.connector.connect(user='root', password='soroush56673sor7',
                                  host='localhost',
                                  database='CommunityPrediction')
    cursor = cnx.cursor()
    date = minDate + pd._libs.tslibs.timestamps.Timedelta(days=Date)
    sqlScript = '''SELECT * From GoldenStandard WHERE date(Creationtimestamp) = ''' + "'" + str(date.date()) + "'"
    cursor.execute(sqlScript)
    result = cursor.fetchall()
    cnx.close()
    return result

def Analyze(DistinctUSers):
    a = []
    b = []
    c = []
    for i in range(len(DistinctUSers)):
        a.append(i)
        b.append(len(UserMentions(DistinctUSers[i][0])))
        c.append(DistinctUSers[i])
    plt.plot(a, b)
    plt.savefig(f'../output/{params.evl["RunId"]}/evl/Mentions_per_User.jpg')
    c = np.asarray(c)
    np.save(f'../output/{params.evl["RunId"]}/evl/users.npy', c)
    return a, b, c

def Analyze2():
    a = []
    b = []
    for i in range(len(DateMentions)):
        a.append(i)
        b.append(len(DateMentions(i)))
    plt.plot(a, b)
    plt.savefig(f'../output/{params.evl["RunId"]}/evl/Mentions_per_Day.jpg')
    return a, b

# def LogFile():
#     file_handler = logging.FileHandler("../logfile.log")
#     logger = logging.getLogger()
#     logger.addHandler(file_handler)
#     logger.setLevel(logging.ERROR)
#     return logger

def main():
    # logger = LogFile()
    cmn.logger.info("\nNewsTopicExtraction.py:\n")
    # def NewsTopics():
    NewsText = TextExtractor()
    NewsText = np.asarray(NewsText)
    data = pd.DataFrame({'Id': NewsText[:, 0], 'Text': NewsText[:, 1]})
    data_text = data['Text']
    cmn.logger.info("len(data) for news extraction query: "+str(len(data_text))+'\n')
    documents = data_text
    processed_docs = []
    for tweet in documents:
        processed_docs.append(tweet.split(','))
    processed_docs = np.asarray(processed_docs)
    DicPath = glob.glob(f'../output/{params.evl["RunId"]}/tml/*topics_TopicModelingDictionary.mm')[0]
    dictionary = gensim.corpora.Dictionary.load(DicPath)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # LDA Model Loading
    model_name = glob.glob(f'../output/{params.evl["RunId"]}/tml/*.model')[0]
    print('model name:', model_name)
    cmn.logger.critical("model "+model_name + " is loaded.")
    GenMal = model_name.split('/')[-1].split('_')[0]
    if GenMal == 'gensim':
        ldaModel = gensim.models.ldamodel.LdaModel.load(model_name)
        print('Lda Model Loaded (Gensim)')
    elif GenMal == 'mallet':
        ldaModel = gensim.models.wrappers.LdaMallet.load(model_name)
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        print('Lda Model Loaded (Mallet)')
    else:
        print('Wrong Library!')


    topics = ldaModel.get_document_topics(bow_corpus)
    cmn.logger.critical("Topics are extracted for news dataset based on the tweets extracted topics.\n")
    topics.save(f'../output/{params.evl["RunId"]}/evl/NewsTopics.mm')
    print(model_name)
    # a,b = Analyze2()
    # plt.close()
    # c,d,e = Analyze()
    # plt.close()

