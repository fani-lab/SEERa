import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 19


general = {
    'Comment': '',
    'RunId': RunID,
}
dal = {
    'path': ['../data/tweets_all.csv', '../data/tagmeannotation_N80000.csv'],
    'userModeling': True,
    'timeModeling': True,
    'start': '2010-11-01',
    'end': '2010-12-14',
    'timeInterval': 1,
    'preProcessing': False,
    'TagME': False,
    'lastRowsNumber': -1
}
tml = {
    'path2saveTML': f'../output/{RunID}/tml',
    'num_topics': 25,
    'library': 'gensim',
    'mallet_home': 'C:/Users/sorou/mallet-2.0.8',
    'filterExtremes': True,
    'JO': False,
    'Bin': True,
    'Threshold': 0.2
}
uml = {
    'UserSimilarityThreshold': 0.0,
    'path2saveUML': f'../output/{RunID}/uml'


}
gel = {
    'path2saveGEL': f'../output/{RunID}/gel',
    'EmbeddingDim': 40,
    'method': 'Node2Vec'
}
evl = {
    'EvaluationType': 'Extrinsic',  # ['Intrinsic', 'Extrinsic']
    # If intrinsic evaluation:
    'EvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure',
                          'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'GoldenStandardPath': '/path2GS',
    'Threshold': 0,
    'TopK': 20
}
