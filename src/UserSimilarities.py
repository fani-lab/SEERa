import warnings
import logging
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import networkx as nx
import time
import pandas as pd
import datetime
import os
import numpy as np
from src.TopicModeling import topic_modeling as TM
from src.UsersGraph import CreateUsersGraph as UG
import glob

# Changing saves location
def ChangeLoc():
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y_%m_%d__%H_%M")
    if not os.path.exists('output'):
        os.mkdir('output')
    os.chdir('output')
    if os.path.exists(dt_string):
        dt_string = dt_string + '_v2'
    os.mkdir(dt_string)
    os.chdir(dt_string)

def LogFile():
    file_handler = logging.FileHandler("logfile.log")
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.ERROR)
    return logger

def Doc2Topic(ldaModel, doc, threshold=0.2, justOne=True, binary=True):
    doc_topic_vector = np.zeros((ldaModel.num_topics))
    try:
        d2tVector = ldaModel.get_document_topics(doc)
    except:
        gen_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        d2tVector = gen_model.get_document_topics(doc)
    a = np.asarray(d2tVector)
    if justOne:
        doc_topic_vector[a[:, 1].argmax()] = 1
    else:
        threshold = a[:, 1].max()
        for i in d2tVector:
            if i[1] >= threshold:
                if binary:
                    doc_topic_vector[i[0]] = 1
                else:
                    doc_topic_vector[i[0]] = i[1]
    doc_topic_vector = np.asarray(doc_topic_vector)
    return doc_topic_vector

def US_main():
    ChangeLoc()
    logger = LogFile()
    print('Topic modeling started')
    topic_num = 75
    TopicModel = TM(num_topics=topic_num, filterExtremes=True, library='gensim', logger=logger)
    logger.critical("Topic modeling finished. Return to UserSimilarities.py\n\n")
    print('Topic modeling finished')
    dictionary, bow_corpus, totalTopics, lda_model, num_topics, processed_docs, documents = TopicModel
    dictionary.save('TopicModelingDictionary.mm')
    # logger.critical("Dictionary saved as TopicModelingDictionary.mm\n")
    end_date = documents['CreationDate'].max()
    end_date = datetime.datetime(2010, 11, 17, 0, 0, 0)
    np.save('end_date.npy', end_date)
    daysBefore = 10
    logger.critical("Run the model for last "+str(daysBefore)+" days.")
    day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=daysBefore)
    logger.critical("From "+str(day.date())+" to "+str(end_date.date())+'\n')
    total_users_topic_interests = []
    all_users = documents['userId']
    logger.critical("All documents:" + str(len(all_users))+'\n')
    all_users_shrinked = pd.core.series.Series(list(set(all_users)))
    logger.critical("All distinct users:" + str(len(all_users_shrinked))+'\n')
    np.save('AllUsers.npy', np.asarray(all_users_shrinked))
    logger.critical("All distinct users are saved as AllUsers.npy with length=" + str(len(all_users_shrinked))+'\n')
    users_topic_interests = np.zeros((len(set(all_users)), topic_num))
    logger.critical("users_topic_interests=" + str(users_topic_interests.shape)+'\n')
    total_user_ids = []
    max_users = 0
    daycounter = 1
    JO = True
    Bin = True
    Threshold = 0.4
    logger.critical('Just one topic? ' + str(JO) + ' /  Binary topic? '+ str(Bin) + ' / Threshold: '+str(Threshold)+'\n')
    lenUsers = []
    while day <= end_date:
        c = documents[(documents['CreationDate'] == day)]
        print(str(len(c)) + ' users has twitted in '+str(day))
        logger.critical(str(len(c)) + ' users has twitted in '+str(day)+'\n')
        texts = c['Text']
        users = c['userId']
        lenUsers.append(len(users))
        users_Ids = []
        for userTextidx in range(min(5000, len(c['Text']))):
            doc = texts.iloc[userTextidx]
            user_bow_corpus = dictionary.doc2bow(doc.split(','))
            D2T = Doc2Topic(lda_model, user_bow_corpus, threshold=Threshold, justOne=JO, binary=Bin)
            users_topic_interests[all_users_shrinked[all_users_shrinked == users.iloc[userTextidx]].index[0]] = D2T
            users_Ids.append(users.iloc[userTextidx])
        total_user_ids.append(users_Ids)
        total_users_topic_interests.append(users_topic_interests)
        day = day + pd._libs.tslibs.timestamps.Timedelta(days=1)
        daycounter += 1
    total_users_topic_interests = np.asarray(total_users_topic_interests)
    graphs = []

    for day in range(len(total_users_topic_interests)):
        if day % 2 == 0 or True:
            print(day, '/', len(total_users_topic_interests))

        daystr = str(day+1)
        if day < 9:
            daystr = '0' + daystr
        np.save('Day' + daystr + 'UsersTopicInterests.npy', total_users_topic_interests[day])
        logger.critical('UsersTopicInterests.npy is saved for day:'+str(day)+ ' with shape: ' + str(total_users_topic_interests[day].shape)+'\n')
        np.save('Day' + daystr + 'UserIDs.npy', users_Ids[day])
        logger.critical('A graph is being created for '+str(day)+' with '+str(len(total_users_topic_interests[day]))+' users\n')

        graph = UG(day, total_users_topic_interests[day])
        graphs.append(graph)
    logger.critical('Number of users per day: '+str(lenUsers)+'\n')
    print('Graph Created!')
    os.mkdir('graphs')
    os.chdir('graphs')
    for i in range(len(graphs)):
        if i < 9:
            nx.write_gpickle(graphs[i], '0'+str(i+1)+'.net')
        else:
            nx.write_gpickle(graphs[i], str(i+1)+'.net')
    logger.critical('Graphs are written in "graphs" directory\n')
