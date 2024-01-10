import datetime
import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import params
from cmn import Common as cmn
from tml import TopicModeling as tm
from uml import UsersGraph as UG

def main(documents, dictionary, lda_model):
    if not os.path.isdir(params.uml['path2save']): os.makedirs(params.uml['path2save'])
    documents['TopicInterests'] = documents['Tokens'].apply(lambda tokens: str(list(tm.doc2topics(lda_model, dictionary.doc2bow(tokens)))))
    if not os.path.isdir(f'{params.uml["path2save"]}/graphs'): os.makedirs(f'{params.uml["path2save"]}/graphs')
    if not os.path.isdir(f'{params.uml["path2save"]}/user_interests'): os.makedirs(f'{params.uml["path2save"]}/user_interests')
    documents.to_csv(f"../output/{params.uml['path2save']}/documents.csv", encoding='utf-8', index=False, header=True)
    start_date = datetime.datetime.strptime(params.dal['start'], '%Y-%m-%d')
    end_date = datetime.datetime.strptime(params.dal['end'], '%Y-%m-%d')
    end_timestamp = (end_date - start_date).days // params.dal['timeInterval']
    len_users = [documents[documents['TimeStamp'] == ts]['UserId'].nunique() for ts in range(end_timestamp)]
    cmn.logger.info(f'UserSimilarity: Number of users per day: {len_users}')
    graphs = UG.graph_generator(documents)
    cmn.logger.info(f'UserSimilarity: Graphs are written in "graphs" directory')
    return graphs

