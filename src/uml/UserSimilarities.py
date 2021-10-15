import datetime, time
import os, sys
import numpy as np
import glob
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import networkx as nx
import pandas as pd

sys.path.extend(["../"])
from cmn import Common as cmn
from dal import DataReader as dr
from dal import DataPreparation as dp
from tml import TopicModeling as tm
from uml import UsersGraph as UG
from cpl import GraphClustering as GC
from cpl import GraphReconstruction_main as GR
import params



def main(start='2010-11-01', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'],
             userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=0,
             num_topics=20, filterExtremes=True, library='gensim', path_2_save_tml=params.uml['path2saveTML'],
             path2_save_uml=params.uml['path2saveUML'], JO=True, Bin=True, Threshold = 0.4, RunId=0):
    if not os.path.isdir(path2_save_uml): os.makedirs(path2_save_uml)
    dataset = dr.load_tweets(Tagme=True, start=start, end=end, stopwords=['www', 'RT', 'com', 'http'])
    processed_docs, documents = dp.data_preparation(dataset, userModeling=userModeling, timeModeling=timeModeling,
                                                    preProcessing=preProcessing, TagME=TagME, lastRowsNumber=lastRowsNumber,
                                                    startDate=params.uml['start'])
    pp = np.asarray(processed_docs)
    cmn.logger.info(f'UserSimilarity: Processed docs shape: {pp.shape}')
    cmn.logger.info(f'UserSimilarity: Topic modeling ...')
    dictionary, bow_corpus, totalTopics, lda_model = tm.topic_modeling(processed_docs, num_topics=num_topics, filterExtremes=filterExtremes, library=library, path_2_save_tml=path_2_save_tml)
    cmn.logger.info(f'UserSimilarity: Topic modeling done')

    # dictionary.save('TopicModelingDictionary.mm') moved to topic_modeling.py

    ##### moved to the main.py.
    ##### we can play with start and end to have same effect
    # end_date = datetime.datetime(2010, 11, 17, 0, 0, 0)
    # np.save('end_date.npy', end_date)
    # daysBefore = 10
    # logger.critical("Run the model for last "+str(daysBefore)+" days.")
    # day = end_date - pd._libs.tslibs.timestamps.Timedelta(days=daysBefore)
    # logger.critical("From "+str(day.date())+" to "+str(end_date.date())+'\n')
    #
    total_users_topic_interests = []
    all_users = documents['userId']
    cmn.logger.info(f'UserSimilarity: All users size {len(all_users)}')
    unique_users = pd.core.series.Series(list(set(all_users)))
    cmn.logger.info(f'UserSimilarity: All distinct users:{len(unique_users)}')
    np.save(f'{path2_save_uml}/AllUsers.npy', np.asarray(unique_users))
    users_topic_interests = np.zeros((len(set(all_users)), num_topics))
    cmn.logger.info(f'UserSimilarity: users_topic_interests={users_topic_interests.shape}')
    total_user_ids = []
    max_users = 0
    daycounter = 1

    cmn.logger.info(f'UserSimilarity: Just one topic? {JO}, Binary topic? {Bin}, Threshold: {Threshold}')
    lenUsers = []

    end_date = documents['CreationDate'].max()
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    day = documents['CreationDate'].min()
    day = datetime.datetime.strptime(start, '%Y-%m-%d')


    ## FOR TEST:
    # day = end_date


    while day <= end_date:
        c = documents[(documents['CreationDate'] == day)]
        cmn.logger.info(f'{len(c)} users has twitted in {day}')
        texts = c['Text']
        users = c['userId']
        lenUsers.append(len(users))
        users_Ids = []
        for userTextidx in range(len(c['Text'])):
        # for userTextidx in range(min(5000, len(c['Text']))):
            doc = texts.iloc[userTextidx]
            user_bow_corpus = dictionary.doc2bow(doc.split(','))
            D2T = tm.doc2topics(lda_model, user_bow_corpus, threshold=Threshold, justOne=JO, binary=Bin)
            users_topic_interests[unique_users[unique_users == users.iloc[userTextidx]].index[0]] = D2T
            users_Ids.append(users.iloc[userTextidx])
        total_user_ids.append(users_Ids)
        total_users_topic_interests.append(users_topic_interests)
        day = day + pd._libs.tslibs.timestamps.Timedelta(days=1)
        daycounter += 1
    total_users_topic_interests = np.asarray(total_users_topic_interests)
    graphs = []

    if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
    if not os.path.isdir(f'{path2_save_uml}/graphs/pajek'): os.makedirs(f'{path2_save_uml}/graphs/pajek')
    for day in range(len(total_users_topic_interests)):
        if day % 2 == 0 or True:
            cmn.logger.info(f'UserSimilarity: {day} / {len(total_users_topic_interests)}')

        daystr = str(day+1)
        if day < 9:
            daystr = '0' + daystr
        np.save(f'{path2_save_uml}/Day{daystr}UsersTopicInterests.npy', total_users_topic_interests[day])
        np.save(f'{path2_save_uml}/Day{daystr}UserIDs.npy', total_user_ids[day])
        cmn.logger.info(f'UserSimilarity: UsersTopicInterests.npy is saved for day:{day} with shape: {total_users_topic_interests[day].shape}')

        graph = UG.create_users_graph(day, total_users_topic_interests[day], f'{path2_save_uml}/graphs/pajek')
        cmn.logger.info(f'UserSimilarity: A graph is being created for {day} with {len(total_users_topic_interests[day])} users')
        # graphs.append(graph)
        nx.write_gpickle(graph, f'{path2_save_uml}/graphs/{daystr}.net')
    cmn.logger.info(f'UserSimilarity: Number of users per day: {lenUsers}')
    cmn.logger.info(f'UserSimilarity: Graphs created!')

    # for i in range(len(graphs)):
    #     if i < 9:
    #         nx.write_gpickle(graphs[i], f'{path2_save_uml}/graphs/0{i+1}.net')
    #     else:
    #         nx.write_gpickle(graphs[i], f'{path2_save_uml}/graphs/{i+1}.net')
    cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
    Communities = GC.main(RunId=RunId)
    GR.main(RunId=RunId)
    return Communities


## test
# main(start='2010-11-10', end='2010-11-17', stopwords=['www', 'RT', 'com', 'http'],
#              userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=0,
#              num_topics=20, filterExtremes=True, library='gensim', path_2_save_tml='../../output/tml',
#              path2_save_uml='../../output/uml', JO=True, Bin=True, Threshold = 0.4)