general = {
    'comment': '',
    'baseline': '19/LDA.RecurrentGCN',
    'cuda': '-1',
    'method': 'PyG'
}
dal = {
    'stopwordPath': 'dal/gist_stopwords.txt',
    'path': '../data/toy',
    'userModeling': True,
    'timeModeling': True,
    'start': '2010-12-01',
    'end': '2010-12-04',
    'timeInterval': 1,  # unit of day,
    'testTimeIntervals': 1, # how many time intervals must be left for prediction
    'preProcessing': True,
    'tagMe': False,
    'addLinks': True,
    'getStat': True,
    'statPath2Save': f'../output/{general["baseline"]}/dal/stat',
}

tml = {
    'path2save': f'../output/{general["baseline"]}/tml',
    'numTopics': 30,
    'library': 'gensim',
    'malletHome': 'C:/Users/Soroush/Desktop/mallet-2.0.8/mallet-2.0.8',
    'filterExtremes': True,
    'justOne': False,
    'binary': False,
    'threshold': 0.5,
    'eval': False,
    'visualization': False,
    'method': 'LDA' #[LDA]
}

uml = {
    'connections': 'Explicit', # can be ['Explicit', 'Implicit']. Read more at Readme
    'userSimilarityThreshold': 0.25,
    'graphType': 'Static', # can be ['Static', 'Dynamic']. Read more at Readme
    'featureType': 'Topics', # can be ['Topics', 'UserInfo', 'UserInfo+Topic']. Read more at Readme
    'path2save': f'../output/{general["baseline"]}/uml'
}

gel = {
    'path2save': f'../output/{general["baseline"]}/gel',
    'embeddingDim': 128,
    'epoch': 1000,
    'method': 'RecurrentGCN', #one of ['RecurrentGCN']
    'pyg_method': 'something'
}

cpl = {
    'path2save': f'../output/{general["baseline"]}/cpl',
    'type': 'matrix_based', # ['matrix_based', 'graph_based']
    'method': 'DBSCAN', # type=='matrix_based':['DBSCAN'] / type='graph_based':['louvain']
    'minSize': 5
}
evl = {
    'topK': 500,
    'evaluationType': 'Extrinsic',  # ['Intrinsic', 'Extrinsic']
    # If Extrinsic evaluation:
    'extrinsicEvaluationMetrics': {'success_1', 'success_5', 'success_10', 'success_100', 'success_500'},
    # If intrinsic evaluation:
    'intrinsicEvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure', 'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'goldenStandardPath': '/path2GS',
    'threshold': 0,


}

apl = {
    'path2save': f'../output/{general["baseline"]}/apl',
    'topK': 500,
    'textTitle': 'Text',
    'crawlURLs': False,
    'stat': True
}