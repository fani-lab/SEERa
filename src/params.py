import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 60

# SQL setting
user = 'root'
password = 'Ghsss.34436673'
host = 'localhost'
database = 'twitter3'


uml = {
    'Comment': 'Corrected - Real test',
    'RunId': RunID,

    'start': '2010-12-17',
    'end': '2010-12-17',
    'lastRowsNumber': 10000,

    'num_topics': 25,
    'library': 'gensim',

    'mallet_home': 'C:/Users/sorou/mallet-2.0.8',

    'userModeling': True,
    'timeModeling': True,
    'preProcessing': False,
    'TagME': False,
     

    'filterExtremes': True,
    'JO': False,
    'Bin': True,
    'Threshold': 0.2,
    'UserSimilarityThreshold': 0.2
}

evl = {
    'RunId': RunID,
    'Threshold': 0,
    'TopK': 20
}
