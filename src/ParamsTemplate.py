import os
import multiprocessing

general = {
    'comment': '',
    'baseline': '@baseline',
    'cuda': '-1'
}
dal = {
    'path': '../data/toy.synthetic',
    'userModeling': True,#if true, timeModeling must be true also
    'timeModeling': True,
    'start': '2010-12-01',
    'end': '2010-12-04',
    'timeInterval': 1,  # unit of day
    'tagMe': False
}
tml = {
    'path2save': f'../output/{general["baseline"]}/tml',
    'numTopics': 3,
    'malletHome': '/Users/sharjeelmustafa/Documents/02 Work/01 Research/Y3-2022-F/SEERa/src/tml/Mallet-202108',
    'filterExtremes': True,
    'justOne': False,
    'binary': False,
    'threshold': 0.3,
    'method': '@tml_method', #['lda.gensim', 'lda.mallet', 'gsdmm'],
    'nCore': multiprocessing.cpu_count(),
    'iterations': 10
}
uml = {
    'userSimilarityThreshold': 0.8,
    'path2save': f'../output/{general["baseline"]}/uml'
}
gel = {
    'path2save': f'../output/{general["baseline"]}/gel',
    'embeddingDim': 64,
    'epoch': 10,
    'method': '@gel_method' #one of ['AE', 'DynAE', 'DynRNN', 'DynAERNN']
}
cpl = {
    'path2save': f'../output/{general["baseline"]}/cpl',
    'method': 'louvain',
    'minSize': 10
}
evl = {
    'evaluationType': 'Extrinsic',  # ['Intrinsic', 'Extrinsic']
    # If Extrinsic evaluation:
    'extrinsicMetrics': {'success', 'ndcg_cut', 'map_cut'},
    # If intrinsic evaluation:
    'intrinsicMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure', 'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'goldenStandardPath': '/path2GS',
    'threshold': 0,


}

apl = {
    'communityBased': False,
    'path2save': f'../output/{general["baseline"]}/apl',
    'topK': 100,
    'textTitle': 'Text'
}