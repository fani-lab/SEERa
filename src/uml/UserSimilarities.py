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
from dal import DataPreparation as dp


def main(documents, dictionary, lda_model, num_topics, path2_save_uml, just_one, binary, threshold):
    if not os.path.isdir(path2_save_uml): os.makedirs(path2_save_uml)
    if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
    if not os.path.isdir(f'{path2_save_uml}/graphs/pajek'): os.makedirs(f'{path2_save_uml}/graphs/pajek')
    total_users_topic_interests = pd.DataFrame()
    all_users = documents['UserId']
    unique_users = all_users.unique()
    users_topic_interests = np.zeros((len(unique_users), num_topics))
    np.save(f'{path2_save_uml}/Users.npy', np.asarray(unique_users))
    len_users = []
    end_date = documents['CreationDate'].max()
    day = documents['CreationDate'].min()
    while day <= end_date:
        users_topic_interests = pd.DataFrame()
        c = documents[(documents['CreationDate'].dt.date == day.date())]
        cmn.logger.info(f'{len(c)} users have twitted in {day.date()}')
        len_users.append(len(c['UserId']))
        for index, row in c.iterrows():
            if Params.dal['tagMe']:
                import tagme
                tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"
                doc = dp.tagme_annotator(row['Text'])
            else:
                doc = row['Text']
            user = row['UserId']
            user_bow_corpus = dictionary.doc2bow(doc.split())
            d2t = tm.doc2topics(lda_model, user_bow_corpus, threshold=threshold, just_one=just_one, binary=binary)
            users_topic_interests[user] = d2t

        #for those with no document, zero padding
        for user in unique_users:
            if user not in users_topic_interests:
                users_topic_interests[user] = np.zeros(d2t.shape)

        np.save(f'{path2_save_uml}/Day{day.date()}UsersTopicInterests.npy', users_topic_interests)
        np.save(f'{path2_save_uml}/Day{day.date()}UserIDs.npy', c['UserId'].values)
        cmn.logger.info(f'UserSimilarity: UsersTopicInterests.npy is saved for day:{day.date()} with shape: {users_topic_interests.T.shape}')

        graph = UG.create_users_graph(day, users_topic_interests, f'{path2_save_uml}/graphs/pajek')
        nx.write_gpickle(graph, f'{path2_save_uml}/graphs/{day.date()}.net')
        cmn.logger.info(f'UserSimilarity: A graph is being created for day: {day.date()} with {len(users_topic_interests.T)} users')
        cmn.logger.info(f'UserSimilarity: Number of users per day: {len_users}')
        cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
        day = day + pd._libs.tslibs.timestamps.Timedelta(days=Params.dal['timeInterval'])
