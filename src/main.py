from shutil import copyfile
import sys, os, glob, pickle
import numpy as np
import pandas as pd
import gensim
import networkx as nx

import params
from cmn import Common as cmn

def run_pipeline():
    copyfile('params.py', f'../output/{params.general["runId"]}/params.py')
    os.environ["CUDA_VISIBLE_DEVICES"] = params.general['cuda']

    cmn.logger.info(f'1. Data Reading & Preparation ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading perprocessed files ...')
        with open(f"../output/{params.general['runId']}/documents.csv", 'rb') as infile: documents = pd.read_csv(infile, parse_dates=['CreationDate'])
        processed_docs = np.load(f"../output/{params.general['runId']}/prosdocs.npz", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        from dal import DataReader as dr, DataPreparation as dp
        cmn.logger.info(f'Loading perprocessed files failed! Generating files ...')
        dataset = dr.load_tweets(f'{params.dal["path"]}/Tweets.csv', params.dal['start'], params.dal['end'], stopwords=['www', 'RT', 'com', 'http'])
        cmn.logger.info(f'dataset.shape: {dataset.shape}')
        cmn.logger.info(f'dataset.keys: {dataset.keys()}')
        dataset = np.asarray(dataset)

        cmn.logger.info(f'Data Preparation ...')
        processed_docs, documents = dp.data_preparation(dataset,
                                                        userModeling=params.dal['userModeling'],
                                                        timeModeling=params.dal['timeModeling'],
                                                        preProcessing=params.dal['preProcessing'],
                                                        TagME=params.dal['tagMe'],
                                                        startDate=params.dal['start'],
                                                        timeInterval=params.dal['timeInterval'])

    cmn.logger.info(f'processed_docs.shape: {processed_docs.shape}')
    cmn.logger.info(f'documents.shape: {documents.shape}')

    cmn.logger.info(f'2. Topic modeling ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading LDA model ...')
        dictionary = gensim.corpora.Dictionary.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['numTopics']}topics_TopicModelingDictionary.mm")
        lda_model = gensim.models.LdaModel.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['numTopics']}topics.model")
    except (FileNotFoundError, EOFError) as e:
        from tml import TopicModeling as tm
        cmn.logger.info(f'Loading LDA model failed! Training LDA model ...')
        dictionary, _, _, lda_model = tm.topic_modeling(processed_docs,
                                                        num_topics=params.tml['numTopics'],
                                                        filter_extremes=params.tml['filterExtremes'],
                                                        library=params.tml['library'],
                                                        path_2_save_tml=params.tml['path2save'])

    cmn.logger.info(f'dictionary.shape: {len(dictionary)}')

    # User Graphs
    cmn.logger.info(f"3. Temporal Graph Creation ...")
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f"Loading users' graph stream ...")
        with open(f'{params.uml["path2save"]}/graphs/graphs.pkl', 'rb') as g: graphs = pickle.load(g)
    except (FileNotFoundError, EOFError) as e:
        from uml import UserSimilarities as US
        cmn.logger.info(f"Loading users' graph stream failed! Generating the stream ...")
        US.main(documents, dictionary, lda_model,
                num_topics=params.tml['numTopics'],
                path2_save_uml=params.uml['path2save'],
                just_one=params.tml['justOne'], binary=params.tml['binary'], threshold=params.tml['threshold'])

        graphs_path = glob.glob(f'{params.uml["path2save"]}/graphs/*.net')
        graphs = []
        for gp in graphs_path: graphs.append(nx.read_gpickle(gp))
        with open(f'{params.uml["path2save"]}/graphs/graphs.pkl', 'wb') as g: pickle.dump(graphs, g)

    # Graph Embedding
    cmn.logger.info(f'4. Temporal Graph Embedding ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading embeddings ...')
        embeddings = np.load(f"{params.gel['path2save']}/embeddings.npz", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        cmn.logger.info(f'Loading embeddings failed! Training ...')
        from gel import graphEmbedding as GE
        embeddings = GE.main(graphs, method=params.gel['method'])

    # Community Extraction
    cmn.logger.info(f'5. Community Prediction ...')
    cmn.logger.info('#' * 50)
    from cpl import GraphClustering as GC
    try:
        cmn.logger.info(f'Loading user clusters ...')
        Communities = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy')
    except:
        cmn.logger.info(f'Loading user clusters failed! Generating user clusters ...')
        Communities = GC.main(embeddings, params.cpl['path2save'], params.cpl['method'])

    # News Article Recommendation
    cmn.logger.info(f'6. Application: News Recommendation ...')
    cmn.logger.info('#' * 50)
    from apl import News
    news_output = News.main()
    return news_output




'''
# Baselines:
gel_baselines = ['AE', 'DynAE', 'DynRNN', 'DynAERNN', 'Node2Vec']
tml_baselines = ['LDA']
current_run_id = params.RunID
for g in gel_baselines:
    params.gel['method'] = g
    for t in tml_baselines:
        current_run_id += 1
        params.tml['method'] = t
        newRunID = f'{current_run_id}__GEL_{g}__TML_{t}'

        with open('params.py') as f:
            lines = f.readlines()

        # Re-open file here
        f2 = open('params.py', 'w')
        for line in lines:
            try:
                if line.split()[0] == "RunID":
                    line = f"RunID = '{newRunID}'\n"
                    f2.write(line)
                else:
                    f2.write(line)
            except:
                f2.write(line)
        f2.close()
        '''
        #import params
if not os.path.isdir(f'../output'): os.makedirs(f'../output')
if not os.path.isdir(f'../output/{params.general["runId"]}'): os.makedirs(
    f'../output/{params.general["runId"]}')
cmn.logger = cmn.LogFile(f'../output/{params.general["runId"]}/log.txt')


c = run_pipeline()
