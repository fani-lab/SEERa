import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 5


general = {
    'comment': '',
    'runId': RunID,
    'cuda': '-1'
}
dal = {
    'path': '../data/toy',
    'userModeling': True,
    'timeModeling': True,
    'start': '2010-12-01',
    'end': '2010-12-04',
    'timeInterval': 1,  # unit of day
    'preProcessing': False,
    'tagMe': False
}
tml = {
    'path2save': f'../output/{RunID}/tml',
    'numTopics': 30,
    'library': 'gensim',
    'malletHome': 'C:/Users/Soroush/Desktop/mallet-2.0.8/mallet-2.0.8',
    'filterExtremes': True,
    'justOne': False,
    'binary': False,
    'threshold': 0.5,
    'method': 'LDA'
}
uml = {
    'userSimilarityThreshold': 0.45,
    'path2save': f'../output/{RunID}/uml'
}
gel = {
    'path2save': f'../output/{RunID}/gel',
    'embeddingDim': 30,
    'epoch': 5,
    'method': 'DynAERNN'
}
cpl = {
    'path2save': f'../output/{RunID}/cpl',
    'method': 'louvain',
    'minSize': 10
}
evl = {
    'topK': 20,
    'evaluationType': 'Extrinsic',  # ['Intrinsic', 'Extrinsic']
    # If Extrinsic evaluation:
    'extrinsicEvaluationMetrics': {'success_1', 'success_5', 'success_10', 'success_100'},
    # If intrinsic evaluation:
    'intrinsicEvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure', 'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'goldenStandardPath': '/path2GS',
    'threshold': 0,


}

apl = {
    'path2save': f'../output/{RunID}/apl',
    'topK': 20,
    'textTitle': 'Text'
}