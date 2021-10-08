import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import glob
import os, sys
import tagme
sys.path.extend(["../"])
import params
from tml import TopicModeling as tm
from cmn import Common as cmn
from dal import DataPreparation as DP


def DistinctUsersandMinMaxDate():
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
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


def TextExtractor(TagME = True, stopwords=['www', 'RT', 'com', 'http']):
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
    cursor = cnx.cursor()
    # sqlScript = '''
    # SELECT NewsId,GROUP_CONCAT('', Word)
    #             FROM NewsTagMeAnnotations
    #             inner join news on  news.Id=NewsTagMeAnnotations.NewsId
    #             group by NewsId
    #             '''

    ## FAKE SQL SCRIPT:
    TagME_SQL = False
    if TagME and TagME_SQL:
        sqlScript = f'''select distinct(T.NewsId), T.GC from (SELECT NewsId, GROUP_CONCAT('', Word) as GC FROM
        twitter3.NewsTagMeAnnotations inner join twitter3.news on twitter3.news.Id = twitter3.NewsTagMeAnnotations.NewsId
        WHERE twitter3.NewsTagMeAnnotations.Score > 0.07 AND NewsTagMeAnnotations.Word NOT IN ("{'","'.join(stopwords)}")
        group by NewsId) as T
        inner join twitter3.goldenstandard2 on twitter3.goldenstandard2.NewsId = T.NewsId
        '''
    else:
        sqlScript = 'select Id,Text from news LIMIT 2000'
    print(sqlScript)

    cursor.execute(sqlScript)
    result = cursor.fetchall()
    cnx.close()
    return np.asarray(result), TagME, TagME_SQL


def UserMentions(User, *args, minDate):
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
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
    cnx = mysql.connector.connect(user=params.user, password=params.password, host=params.host, database=params.database)
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

def TAGME(text, threshold=0.05):
    global DataLen
    annotations = tagme.annotate(text)
    result = []
    if annotations is not None:
        for keyword in annotations.get_annotations(threshold):
            result.append(keyword.entity_title)
    return result
def main():
    cmn.logger.info("\nNewsTopicExtraction.py:\n")
    NewsIds_NewsText, TagME, TagME_SQL = TextExtractor()
    print(NewsIds_NewsText.shape)
    NewsIds = NewsIds_NewsText[:, 0]
    # cmn.save2excel(NewsIds, 'evl/NewsIds')
    # np.save(f'../output/{params.evl["RunId"]}/evl/NewsIds.npy', NewsIds)
    if (TagME and TagME_SQL) or (not TagME):
        NewsText = NewsIds_NewsText[:, 1]
    if TagME:
        NewsText_temp = pd.Series(NewsIds_NewsText[:, 1])
        print(type(NewsText_temp))
        NewsText = NewsText_temp.map(TAGME)
    cmn.save2excel(NewsText, 'evl/NewsText')
    data = pd.DataFrame({'Id': NewsIds, 'Text': NewsText})
    data_text = data['Text']
    cmn.logger.info("len(data) for news extraction query: "+str(len(data_text))+'\n')
    documents = data_text
    processed_docs = []
    for tweet in documents:
        processed_docs.append(tweet.split(','))
    processed_docs = np.asarray(processed_docs)
    DicPath = glob.glob(f'../output/{params.evl["RunId"]}/tml/*topics_TopicModelingDictionary.mm')[0]
    dictionary = gensim.corpora.Dictionary.load(DicPath)
    ## bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # LDA Model Loading
    model_name = glob.glob(f'../output/{params.evl["RunId"]}/tml/*.model')[0]
    print('model name:', model_name)
    cmn.logger.critical("model "+model_name + " is loaded.")
    GenMal = model_name.split('\\')[-1].split('_')[0]
    if GenMal == 'gensim':
        ldaModel = gensim.models.ldamodel.LdaModel.load(model_name)
        print('Lda Model Loaded (Gensim)')
    elif GenMal == 'mallet':
        ldaModel = gensim.models.wrappers.LdaMallet.load(model_name)
        ldaModel = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        print('Lda Model Loaded (Mallet)')
    else:
        print('Wrong Library!')


    # topics = ldaModel.get_document_topics(bow_corpus)
    totalNewsTopics = []
    for news in range(len(processed_docs)):
        news_bow_corpus = dictionary.doc2bow(processed_docs[news])
        topics = tm.doc2topics(ldaModel, news_bow_corpus, threshold=params.evl['Threshold'], justOne=params.uml['JO'],
                               binary=params.uml['Bin'])
        totalNewsTopics.append(topics)

    cmn.logger.critical("Topics are extracted for news dataset based on the tweets extracted topics.\n")
    totalNewsTopics = np.asarray(totalNewsTopics)
    cmn.save2excel(totalNewsTopics, 'evl/totalNewsTopics')
    np.save(f'../output/{params.evl["RunId"]}/evl/NewsTopics.npy', totalNewsTopics)
    print('/////////////////////////////')
    print(model_name)
    print(NewsIds.shape)
    print(totalNewsTopics.shape)
    print('/////////////////////////////')
    # a,b = Analyze2()
    # plt.close()
    # c,d,e = Analyze()
    # plt.close()

