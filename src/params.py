import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 64

# SQL setting
user = 'root'
password = 'Ghsss.34436673'
host = 'localhost'
database = 'twitter3'


uml = {
    'Comment': '',
    'RunId': RunID,

    'start': '2010-12-17',
    'end': '2010-12-17',
    'timeInterval': 1,
    'lastRowsNumber': 300,

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
    'UserSimilarityThreshold': 0.2,
    'GraphEmbedding': 'Node2Vec',
    'path2saveUML': f'../output/{RunID}/uml',
    'path2saveTML': f'../output/{RunID}/tml',
    'path2saveGEL': f'../output/{RunID}/gel',
    'EmbeddingDim': 40
}

evl = {
    'EvaluationType': 'Extrinsic', # ['Intrinsic', 'Extrinsic']

    # If intrinsic evaluation:
    'EvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure',
                          'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'RunId': RunID,
    'Threshold': 0,
    'TopK': 20
}
