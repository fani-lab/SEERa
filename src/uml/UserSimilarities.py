import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
import pandas as pd

import Params
from cmn import Common as cmn
from tml import TopicModeling as tm
from uml import UsersGraph as UG


def main(documents, dictionary, lda_model, num_topics, path2_save_uml, just_one, binary, threshold):
    if not os.path.isdir(path2_save_uml): os.makedirs(path2_save_uml)
    if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
    if not os.path.isdir(f'{path2_save_uml}/graphs/pajek'): os.makedirs(f'{path2_save_uml}/graphs/pajek')
    total_users_topic_interests = pd.DataFrame()
    all_users = documents['userId']
    cmn.logger.info(f'UserSimilarity: All users size {len(all_users)}')
    unique_users = pd.core.series.Series(list(set(all_users)))
    cmn.logger.info(f'UserSimilarity: All distinct users:{len(unique_users)}')
    np.save(f'{path2_save_uml}/users.npy', np.asarray(unique_users))
    users_topic_interests = np.zeros((len(unique_users), num_topics))
    cmn.logger.info(f'UserSimilarity: users_topic_interests={users_topic_interests.shape}')
    cmn.logger.info(f'UserSimilarity: Just one topic? {just_one}, Binary topic? {binary}, Threshold: {threshold}')
    len_users = []
    end_date = documents['CreationDate'].max()
    day = documents['CreationDate'].min()
    while day <= end_date:
        users_topic_interests = pd.DataFrame()
        c = documents[(documents['CreationDate'].dt.date == day.date())]
        cmn.logger.info(f'{len(c)} users have twitted in {day}')
        len_users.append(len(c['userId']))
        for index, row in c.iterrows():
            doc = row['Tokens']
            user = row['userId']
            user_bow_corpus = dictionary.doc2bow(doc.split())
            d2t = tm.doc2topics(lda_model, user_bow_corpus, threshold=threshold, just_one=just_one, binary=binary)
            print(doc)
            print(d2t)
            users_topic_interests[user] = d2t
        for users in unique_users:
            if users not in users_topic_interests:
                users_topic_interests[users] = np.zeros(d2t.shape)
        cmn.logger.info(f'UserSimilarity: {day} / {len(total_users_topic_interests)}')
        day_str = str(day.date())
        print("SALAAAM",users_topic_interests)
        np.save(f'{path2_save_uml}/Day{day_str}UsersTopicInterests.npy', users_topic_interests)
        users_topic_interests.to_pickle(f'{path2_save_uml}/Day{day_str}UsersTopicInterests.pkl')
        users_topic_interests.to_csv(f'{path2_save_uml}/Day{day_str}UsersTopicInterests.csv')
        np.save(f'{path2_save_uml}/Day{day_str}UserIDs.npy', c['userId'].values)
        cmn.logger.info(f'UserSimilarity: UsersTopicInterests.npy is saved for day:{day} with shape: {users_topic_interests.shape}')
        graph = UG.create_users_graph(day, users_topic_interests, f'{path2_save_uml}/graphs/pajek')
        cmn.logger.info(f'UserSimilarity: A graph is being created for day:{day} with {len(users_topic_interests)} users')
        nx.write_gpickle(graph, f'{path2_save_uml}/graphs/{day_str}.net')
        cmn.logger.info(f'UserSimilarity: Number of users per day: {len_users}')
        cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
        day = day + pd._libs.tslibs.timestamps.Timedelta(days=1)
