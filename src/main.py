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
