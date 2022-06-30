import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
import pandas as pd

from cmn import Common as cmn
from tml import TopicModeling as tm
from uml import UsersGraph as UG
import params


def main(documents, dictionary, lda_model, num_topics, path2_save_uml, JO, Bin, Threshold):
    if not os.path.isdir(path2_save_uml): os.makedirs(path2_save_uml)
    total_users_topic_interests = []
    all_users = documents['userId']
    cmn.logger.info(f'UserSimilarity: All users size {len(all_users)}')
    unique_users = pd.core.series.Series(list(set(all_users)))
    cmn.logger.info(f'UserSimilarity: All distinct users:{len(unique_users)}')
    np.save(f'{path2_save_uml}/users.npy', np.asarray(unique_users))
    users_topic_interests = np.zeros((len(unique_users), num_topics))
    cmn.logger.info(f'UserSimilarity: users_topic_interests={users_topic_interests.shape}')
    total_user_ids = []
    daycounter = 1
    cmn.logger.info(f'UserSimilarity: Just one topic? {JO}, Binary topic? {Bin}, Threshold: {Threshold}')
    lenUsers = []
    end_date = datetime.datetime.strptime(params.dal['end'], '%Y-%m-%d')
    day = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d')

    while day <= end_date:
        users_topic_interests = np.zeros((len(unique_users), num_topics))
        #users_topic_interests[0] += 1
        c = documents[(documents['CreationDate'].dt.date == day.date())]
        cmn.logger.info(f'{len(c)} users have twitted in {day}')
        texts = c['Tokens']
        users = c['userId']
        lenUsers.append(len(users))
        users_Ids = []
        for userTextidx in range(len(c['Tokens'])):
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

    if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
    if not os.path.isdir(f'{path2_save_uml}/graphs/pajek'): os.makedirs(f'{path2_save_uml}/graphs/pajek')
    for day in range(len(total_users_topic_interests)):
        if day % 2 == 0 or True:
            cmn.logger.info(f'UserSimilarity: {day} / {len(total_users_topic_interests)}')

        daystr = str(day+1)
        np.save(f'{path2_save_uml}/Day{daystr}UsersTopicInterests.npy', total_users_topic_interests[day])
        np.save(f'{path2_save_uml}/Day{daystr}UserIDs.npy', total_user_ids[day])
        cmn.logger.info(f'UserSimilarity: UsersTopicInterests.npy is saved for day:{day} with shape: {total_users_topic_interests[day].shape}')

        graph = UG.create_users_graph(day, total_users_topic_interests[day], f'{path2_save_uml}/graphs/pajek')
        cmn.logger.info(f'UserSimilarity: A graph is being created for day:{day} with {len(total_users_topic_interests[day])} users')
        nx.write_gpickle(graph, f'{path2_save_uml}/graphs/{daystr}.net')
    cmn.logger.info(f'UserSimilarity: Number of users per day: {lenUsers}')
    cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')

def userTopics(documents, total_users_topic_interests, total_user_ids, userid, unique_users, lda_model):
    print(userid)
    idx = unique_users[unique_users == userid].index[0]
    print(idx)
    print(documents['Tokens'][documents['userId'] == userid])
    print(total_users_topic_interests[idx])
    for i, j in enumerate(total_users_topic_interests[idx]):
        if j > 0:
            print(lda_model.print_topic(i))
## test
# main(start='2010-11-10', end='2010-11-17', stopwords=['www', 'RT', 'com', 'http'],
#              userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=0,
#              num_topics=20, filterExtremes=True, library='gensim', path_2_save_tml='../../output/tml',
#              path2_save_uml='../../output/uml', JO=True, Bin=True, Threshold = 0.4)