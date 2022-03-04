from shutil import copyfile
import sys, os
import numpy as np
#sys.path.extend(["../"])
import params
from cmn import Common as cmn
if not os.path.isdir(f'../output'): os.makedirs(f'../output')
if not os.path.isdir(f'../output/{params.general["RunId"]}'): os.makedirs(f'../output/{params.general["RunId"]}')
cmn.logger=cmn.LogFile(f'../output/used_params_runid_{params.general["RunId"]}.log')



def RunPipeline():
    copyfile('params.py', f'../output/used_params_runid_{params.general["RunId"]}.py')


    # Data Reading
    from dal import DataReader as dr, DataPreparation as dp
    cmn.logger.info(f'Data Reading ...')
    dataset = dr.load_tweets(params.dal['path'], params.dal['start'], params.dal['end'], stopwords=['www', 'RT', 'com', 'http'], tagme_threshold=0.07)
    # dataset: Index(['TweetId', 'CreationDate', 'UserId', 'ModificationTimestamp', 'Tokens'], dtype='object')
    cmn.logger.info(f'dataset.shape: {dataset.shape}')
    cmn.logger.info(f'dataset.keys: {dataset.keys()}')
    dataset = np.asarray(dataset)


    # Data Preparation
    cmn.logger.info(f'Data Preparation ...')
    processed_docs, documents = dp.data_preparation(dataset, userModeling=params.dal['userModeling'], timeModeling=params.dal['timeModeling'],
                                                                     preProcessing=params.dal['preProcessing'], TagME=params.dal['TagME'], lastRowsNumber=params.dal['lastRowsNumber'],
                                                                     startDate=params.dal['start'], timeInterval=params.dal['timeInterval'])
    cmn.logger.info(f'processed_docs.shape: {processed_docs.shape}')
    cmn.logger.info(f'documents.shape: {documents.shape}')
    #print(documents.keys())


    # Topic Modeling
    from tml import TopicModeling as tm
    cmn.logger.info(f'Topic modeling ...')
    dictionary, bow_corpus, totalTopics, lda_model = tm.topic_modeling(processed_docs, num_topics=params.tml['num_topics'],
                                                                       filterExtremes=params.tml['filterExtremes'], library=params.tml['library'],
                                                                       path_2_save_tml=params.tml['path2saveTML'])
    cmn.logger.info(f'dictionary.shape: {len(dictionary)}')
    cmn.logger.info(f'bow_corpus.shape: {len(bow_corpus)}')
    cmn.logger.info(f'totalTopics: {totalTopics}')


    # User Graphs
    from uml import UserSimilarities as US
    cmn.logger.info(f"Users' graph generating ...")
    US.main(documents, dictionary, lda_model, num_topics=params.tml['num_topics'], path2_save_uml=params.uml['path2saveUML'],
            JO=params.tml['JO'], Bin=params.tml['Bin'], Threshold=params.tml['Threshold'], RunId=params.general['RunId'])


    # Graph Embedding
    from gel import graphEmbedding as GE
    GE.main(method=params.gel['method'])

    # Community Extraction
    from cpl import GraphClustering as GC, GraphReconstruction_main as GR
    GR.main(RunId=params.general['RunId'])
    Communities = GC.main(RunId=params.general['RunId'])

    return Communities






c = RunPipeline()

'''

    Communities = uml.main(start=params.uml['start'],
             end=params.uml['end'],
             stopwords=['www', 'RT', 'com', 'http'],
             userModeling=params.uml['userModeling'],
             timeModeling=params.uml['timeModeling'],
             preProcessing=params.uml['preProcessing'],
             TagME=params.uml['TagME'],
             lastRowsNumber=params.uml['lastRowsNumber'], #10000, #all rows = 0
             num_topics=params.uml['num_topics'],
             filterExtremes=params.uml['filterExtremes'],
             library=params.uml['library'],
             path_2_save_tml=f'../output/{params.uml["RunId"]}/tml',
             path2_save_uml=f'../output/{params.uml["RunId"]}/uml',
             JO=params.uml['JO'],
             Bin=params.uml['Bin'],
             Threshold=params.uml['Threshold'],
             RunId=params.uml['RunId'])
    GE.main()
    if params.evl['EvaluationType'] == 'Extrinsic':
        result = evl.main(RunId=params.evl['RunId'],
                 path2_save_evl=f'../output/{params.evl["RunId"]}/evl',)
    elif params.evl['EvaluationType'] == 'Intrinsic':
        result = evl.intrinsic_evaluation(Communities, params.evl['GoldenStandardPath'])
    else:
        result = -1
        print('Wrong Evaluation Type. ')
    return result

PytrecResult = RunPipeline()

'''