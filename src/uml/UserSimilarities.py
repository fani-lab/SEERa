import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import networkx as nx
import pandas as pd

import params
from cmn import Common as cmn
from tml import TopicModeling as tm
from uml import UsersGraph as UG

def main(documents, dictionary, lda_model):
    if not os.path.isdir(params.uml['path2save']): os.makedirs(params.uml['path2save'])

    documents['TopicInterests'] = pd.Series
    for index, row in documents.iterrows():
        user_bow_corpus = dictionary.doc2bow(row['Tokens'])
        documents.loc[index, 'TopicInterests'] = str(list(tm.doc2topics(lda_model, user_bow_corpus)))
    if not os.path.isdir(f'{params.uml["path2save"]}/graphs'): os.makedirs(f'{params.uml["path2save"]}/graphs')
    if not os.path.isdir(f'{params.uml["path2save"]}/user_interests'): os.makedirs(f'{params.uml["path2save"]}/user_interests')
    end_date = datetime.datetime.strptime(params.dal['end'], '%Y-%m-%d')
    day = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d')
    startTimeStamp = 0
    endTimeStamp = (end_date - day).days // params.dal['timeInterval']
    timeStamp = startTimeStamp
    len_users = []
    connections_path = f'{params.dal["path"]}/user_connections.csv'
    while timeStamp <= endTimeStamp:
        cmn.logger.info(f'{(documents["TimeStamp"] == timeStamp).sum()} users have twitted in {day+datetime.timedelta(timeStamp)}')
        len_users.append(documents[documents['TimeStamp'] == timeStamp]['UserId'].nunique())
        timeStamp += 1
    cmn.logger.info(f'UserSimilarity: Number of users per day: {len_users}')

    graphs = UG.graph_generator(documents, connections_path)
    cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
    return graphs

