import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import pandas as pd
from tqdm import tqdm
from scipy import sparse
import tagme
import bitermplus as btm

import Params
from cmn import Common as cmn
from tml import TopicModeling as tm
from uml import UsersGraph as UG
from dal import DataPreparation as dp


def main(documents, dictionary, lda_model, path2_save_uml, just_one, binary, threshold):
    if not os.path.isdir(path2_save_uml): os.makedirs(path2_save_uml)
    if not os.path.isdir(f'{path2_save_uml}/graphs'): os.makedirs(f'{path2_save_uml}/graphs')
    all_users = documents['UserId']
    unique_users = all_users.unique()
    np.save(f'{path2_save_uml}/Users.npy', np.asarray(unique_users))
    end_date = documents['CreationDate'].max()
    day = documents['CreationDate'].min()
    while day <= end_date:
        users_topic_interests = pd.DataFrame()
        c = documents[(documents['CreationDate'].dt.date == day.date())]
        cmn.logger.info(f'{len(c)} users have twitted in {day.date()}')
        if Params.tml['method'].lower() == 'btm': d2t_vector = lda_model.transform(btm.get_vectorized_docs(c['Text'], dictionary))
        user_idx = 0
        for index, row in tqdm(c.iterrows(), total=c.shape[0]):
            if Params.tml['method'].lower() == 'btm':
                d2t = d2t_vector[user_idx]
                users_topic_interests[row['UserId']] = d2t
                user_idx += 1
                continue
            if Params.dal['tagMe']:
                tagme.GCUBE_TOKEN = "7d516eaf-335b-4676-8878-4624623d67d4-843339462"
                doc = dp.tagme_annotator(row['Text'])
            else:
                doc = row['Text']
            if Params.tml['method'].lower() == 'btm': d2t = tm.doc2topics(lda_model, [doc], threshold=threshold, just_one=just_one, binary=binary, dic=dictionary)
            else: d2t = tm.doc2topics(lda_model, dictionary.doc2bow(doc.split()), threshold=threshold, just_one=just_one, binary=binary)
            users_topic_interests[row['UserId']] = d2t

        #for those with no document, zero padding
        padded_uti = np.zeros((len(unique_users), d2t.shape[0]))
        padded_uti = pd.DataFrame(padded_uti, index=unique_users).T
        for user in tqdm(users_topic_interests):
            padded_uti[user] = users_topic_interests[user]
        # users_topic_interests = sparse.csr_matrix(padded_uti)
        users_topic_interests = padded_uti
        pd.to_pickle(users_topic_interests, f'{path2_save_uml}/Day{day.date()}UsersTopicInterests.pkl')
        # sparse.save_npz(f'{path2_save_uml}/Day{day.date()}UsersTopicInterests.npz', users_topic_interests)
        pd.to_pickle(c['UserId'], f'{path2_save_uml}/Day{day.date()}UserIDs.pkl')
        cmn.logger.info(f'UserSimilarity: UsersTopicInterests.npz is saved for day:{day.date()} with shape: {users_topic_interests.T.shape}')

        cmn.logger.info(f'UserSimilarity: A graph is being created for day: {day.date()} with {users_topic_interests.shape[1]} users')
        cmn.logger.info(f'UserSimilarity: Number of users per day: {len(c)}')
        cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
        UG.create_users_graph(day, users_topic_interests, path2_save_uml)
        day = day + pd._libs.tslibs.timestamps.Timedelta(days=Params.dal['timeInterval'])
