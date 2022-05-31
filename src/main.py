from shutil import copyfile
import sys, os, glob, pickle
import numpy as np
import pandas as pd
import gensim
import networkx as nx
#sys.path.extend(["../"])

import params
from cmn import Common as cmn

if not os.path.isdir(f'../output'): os.makedirs(f'../output')
if not os.path.isdir(f'../output/{params.general["RunId"]}'): os.makedirs(f'../output/{params.general["RunId"]}')
cmn.logger=cmn.LogFile(f'../output/{params.general["RunId"]}/log.txt')

def RunPipeline():
    copyfile('params.py', f'../output/{params.general["RunId"]}/params.py')
    os.environ["CUDA_VISIBLE_DEVICES"] = params.general['cuda']

    cmn.logger.info(f'1. Data Reading & Preparation ...')
    try:
        cmn.logger.info(f'Loading perprocessed files ...')
        with open(f"../output/{params.general['RunId']}/documents.csv", 'rb') as infile: documents = pd.read_csv(infile, parse_dates=['CreationDate'])
        processed_docs = np.load(f"../output/{params.general['RunId']}/prosdocs.npz", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        from dal import DataReader as dr, DataPreparation as dp
        cmn.logger.info(f'Loading perprocessed files failed! Generating files ...')
        dataset = dr.load_tweets(params.dal['path'], params.dal['start'], params.dal['end'], stopwords=['www', 'RT', 'com', 'http'])
        cmn.logger.info(f'dataset.shape: {dataset.shape}')
        cmn.logger.info(f'dataset.keys: {dataset.keys()}')
        dataset = np.asarray(dataset)

        cmn.logger.info(f'Data Preparation ...')
        processed_docs, documents = dp.data_preparation(dataset,
                                                        userModeling=params.dal['userModeling'],
                                                        timeModeling=params.dal['timeModeling'],
                                                        preProcessing=params.dal['preProcessing'],
                                                        TagME=params.dal['TagME'],
                                                        startDate=params.dal['start'],
                                                        timeInterval=params.dal['timeInterval'])

    cmn.logger.info(f'processed_docs.shape: {processed_docs.shape}')
    cmn.logger.info(f'documents.shape: {documents.shape}')

    cmn.logger.info(f'2. Topic modeling ...')
    try:
        cmn.logger.info(f'Loading LDA model ...')
        dictionary = gensim.corpora.Dictionary.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['num_topics']}topics_TopicModelingDictionary.mm")
        lda_model = gensim.models.LdaModel.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['num_topics']}topics.model")
    except (FileNotFoundError, EOFError) as e:
        from tml import TopicModeling as tm
        cmn.logger.info(f'Loading LDA model failed! Training LDA model ...')
        dictionary, _, _, lda_model = tm.topic_modeling(processed_docs,
                                                        num_topics=params.tml['num_topics'],
                                                        filterExtremes=params.tml['filterExtremes'],
                                                        library=params.tml['library'],
                                                        path_2_save_tml=params.tml['path2save'])

    cmn.logger.info(f'dictionary.shape: {len(dictionary)}')

    # User Graphs
    cmn.logger.info(f"3. Temporal Graph Creation ...")
    try:
        cmn.logger.info(f"Loading users' graph stream ...")
        with open(f'{params.uml["path2save"]}/graphs/graphs.pkl', 'rb') as g: graphs = pickle.load(g)
    except (FileNotFoundError, EOFError) as e:
        from uml import UserSimilarities as US
        cmn.logger.info(f"Loading users' graph stream failed! Generating the stream ...")
        US.main(documents, dictionary, lda_model,
                num_topics=params.tml['num_topics'],
                path2_save_uml=params.uml['path2save'],
                JO=params.tml['JO'], Bin=params.tml['Bin'], Threshold=params.tml['Threshold'])

        graphs_path = glob.glob(f'{params.uml["path2save"]}/graphs/*.net')
        graphs = []
        for gp in graphs_path: graphs.append(nx.read_gpickle(gp))
        with open(f'{params.uml["path2save"]}/graphs/graphs.pkl', 'wb') as g: pickle.dump(graphs, g)

    # Graph Embedding
    cmn.logger.info(f'4. Temporal Graph Embedding ...')
    try:
        cmn.logger.info(f'Loading embeddings ...')
        embeddings = np.load(f"{params.gel['path2save']}/embeddeds.npy", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        cmn.logger.info(f'Loading embeddings failed! Training ...')
        from gel import graphEmbedding as GE
        GE.main(graphs, method=params.gel['method'])

    # Community Extraction
    from cpl import GraphClustering as GC, GraphReconstruction_main as GR
    GR.main(RunId=params.general['RunId'])
    Communities = GC.main(RunId=params.general['RunId'])
    return Communities

c = RunPipeline()
