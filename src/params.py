import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 18

uml = {
    'RunId': RunID,

    'start': '2010-12-20',
    'end': '2010-12-30',
    'lastRowsNumber': 80000,

    'num_topics': 50,
    'library': 'mallet',
    'mallet_home': 'C:/Users/sorou/mallet-2.0.8',

    'userModeling': True,
    'timeModeling': True,
    'preProcessing': False,
    'TagME': False,

    'filterExtremes': True,
    'JO': True,
    'Bin': True,
    'Threshold': 0.4
}

evl = {
    'RunId': RunID,
    'TopK': 10
}