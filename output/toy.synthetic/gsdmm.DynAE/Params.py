
general = {
    'comment': '',
    'baseline': 'toy.syntheticgsdmm11/gsdmm.DynAE',
    'cuda': '-1'
}
dal = {
    'path': '../data/toy.synthetic',
    'userModeling': True,
    'timeModeling': True,
    'start': '2010-12-01',
    'end': '2010-12-04',
    'timeInterval': 1,  # unit of day
    'preProcessing': False,
    'tagMe': False
}
tml = {
    'path2save': f'../output/{general["baseline"]}/tml',
    'library': 'gensim',
    'numTopics': 3,
    'malletHome': 'C:/Users/Soroush/Desktop/mallet-2.0.8/mallet-2.0.8',
    'filterExtremes': False,
    'justOne': False,
    'binary': False,
    'threshold': 0.5,
    'method': 'gsdmm' #[LDA]
}
uml = {
    'userSimilarityThreshold': 0.2,
    'path2save': f'../output/{general["baseline"]}/uml'
}
gel = {
    'path2save': f'../output/{general["baseline"]}/gel',
    'embeddingDim': 32,
    'epoch': 100,
    'method': 'DynAE' #one of ['AE', 'DynAE', 'DynRNN', 'DynAERNN']
}
cpl = {
    'path2save': f'../output/{general["baseline"]}/cpl',
    'method': 'louvain',
    'minSize': 10
}
evl = {
    'evaluationType': 'Extrinsic',  # ['Intrinsic', 'Extrinsic']
    # If Extrinsic evaluation:
    'extrinsicEvaluationMetrics': {'success_1', 'success_5', 'success_10', 'success_100'},
    # If intrinsic evaluation:
    'intrinsicEvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure', 'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'goldenStandardPath': '/path2GS',
    'threshold': 0,


}

apl = {
    'path2save': f'../output/{general["baseline"]}/apl',
    'topK': 20,
    'textTitle': 'Text'
}

gsdmm = {
    "dataset": "News", # selected dataset from data folder
    "timefil": "timefil",
    "MaxBatch": 5,  # The number of saved batches + 1
    "AllBatchNum": 1,  # The number of batches you want to divide the dataset to
    "alpha": 0.03,  # document-topic density
    "beta": 0.03,  # topic-word density
    "iterNum": 5,  # number of iterations
    "sampleNum": 1,
    "wordsInTopicNum": 10,
    "K": 0
}