import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 5


general = {
    'Comment': '',
    'RunId': RunID,
    'cuda': '-1'
}
dal = {
    'path': '../data/toy/Tweets.csv',
    'userModeling': True,
    'timeModeling': True,
    'start': '2010-12-01',
    'end': '2010-12-04',
    'timeInterval': 1, #unit of day
    'preProcessing': False,
    'TagME': False
}
tml = {
    'path2save': f'../output/{RunID}/tml',
    'num_topics': 30,
    'library': 'gensim',
    'mallet_home': 'C:/Users/Soroush/Desktop/mallet-2.0.8/mallet-2.0.8',
    'filterExtremes': True,
    'JO': False,
    'Bin': False,
    'Threshold': 0.5
}
uml = {
    'UserSimilarityThreshold': 0.45,
    'path2save': f'../output/{RunID}/uml'
}
gel = {
    'path2save': f'../output/{RunID}/gel',
    'EmbeddingDim': 30,
    'epoch': 5,
    'method': 'DynAERNN'
}
cpl = {
    'path2save': f'../output/{RunID}/cpl',
    'method': 'louvain',
    'min_size': 10
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

apl = {
    'path2read': f'../data/toy',
    'path2save': f'../output/{RunID}/apl',
    'TopK': 20,
    'Text_Title': 'Title'
}