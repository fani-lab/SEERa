import random
import numpy as np

random.seed(0)
np.random.seed(0)

uml = {
    'runid': 2,

    'start': '2010-11-10',
    'end': '2010-11-17',

    'num_topics': 75,
    'library': 'gensim',
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