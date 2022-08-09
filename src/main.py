from shutil import copyfile
import sys, os, glob, pickle, time, argparse, importlib, traceback

import numpy as np
import pandas as pd
import gensim
import networkx as nx

import Params
from cmn import Common as cmn

def main():
    if not os.path.isdir(f'../output/{Params.general["baseline"]}'): os.makedirs(f'../output/{Params.general["baseline"]}')
    copyfile('Params.py', f'../output/{Params.general["baseline"]}/Params.py')

    os.environ["CUDA_VISIBLE_DEVICES"] = params.general['cuda']

    cmn.logger.info(f'1. Data Reading & Preparation ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading perprocessed files ...')
        with open(f"../output/{Params.general['baseline']}/documents.csv", 'rb') as infile: documents = pd.read_csv(infile, parse_dates=['CreationDate'])
        processed_docs = np.load(f"../output/{Params.general['baseline']}/prosdocs.npz", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        from dal import DataReader as dr, DataPreparation as dp
        cmn.logger.info(f'Loading perprocessed files failed! Generating files ...')
        dataset = dr.load_tweets(f'{Params.dal["path"]}/Tweets.csv', Params.dal['start'], Params.dal['end'], stopwords=['www', 'RT', 'com', 'http'])
        cmn.logger.info(f'dataset.shape: {dataset.shape}')
        cmn.logger.info(f'dataset.keys: {dataset.keys()}')
        dataset = np.asarray(dataset)

        cmn.logger.info(f'Data Preparation ...')
        processed_docs, documents = dp.data_preparation(dataset,
                                                        userModeling=Params.dal['userModeling'],
                                                        timeModeling=Params.dal['timeModeling'],
                                                        preProcessing=Params.dal['preProcessing'],
                                                        TagME=Params.dal['tagMe'],
                                                        startDate=Params.dal['start'],
                                                        timeInterval=Params.dal['timeInterval'])

    cmn.logger.info(f'processed_docs.shape: {processed_docs.shape}')
    cmn.logger.info(f'documents.shape: {documents.shape}')

    cmn.logger.info(f'2. Topic modeling ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading LDA model ...')
        dictionary = gensim.corpora.Dictionary.load(f"{Params.tml['path2save']}/{Params.tml['library']}_{Params.tml['numTopics']}topics_TopicModelingDictionary.mm")
        lda_model = gensim.models.LdaModel.load(f"{Params.tml['path2save']}/{Params.tml['library']}_{Params.tml['numTopics']}topics.model")
    except (FileNotFoundError, EOFError) as e:
        from tml import TopicModeling as tm
        cmn.logger.info(f'Loading LDA model failed! Training LDA model ...')
        dictionary, _, _, lda_model = tm.topic_modeling(processed_docs,
                                                        method=Params.tml['method'],
                                                        num_topics=Params.tml['numTopics'],
                                                        filter_extremes=Params.tml['filterExtremes'],
                                                        library=Params.tml['library'],
                                                        path_2_save_tml=Params.tml['path2save'])
        return lda_model

    cmn.logger.info(f'dictionary.shape: {len(dictionary)}')
    
    # User Graphs
    cmn.logger.info(f"3. Temporal Graph Creation ...")
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f"Loading users' graph stream ...")
        with open(f'{Params.uml["path2save"]}/graphs/graphs.pkl', 'rb') as g: graphs = pickle.load(g)
    except (FileNotFoundError, EOFError) as e:
        from uml import UserSimilarities as US
        cmn.logger.info(f"Loading users' graph stream failed! Generating the stream ...")
        US.main(documents, dictionary, lda_model,
                num_topics=Params.tml['numTopics'],
                path2_save_uml=Params.uml['path2save'],
                just_one=Params.tml['justOne'], binary=Params.tml['binary'], threshold=Params.tml['threshold'])

        graphs_path = glob.glob(f'{Params.uml["path2save"]}/graphs/*.net')
        graphs = []
        for gp in graphs_path: graphs.append(nx.read_gpickle(gp))
        with open(f'{Params.uml["path2save"]}/graphs/graphs.pkl', 'wb') as g: pickle.dump(graphs, g)

    # Graph Embedding
    cmn.logger.info(f'4. Temporal Graph Embedding ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading embeddings ...')
        #embeddings = np.load(f"{Params.gel['path2save']}/embeddings.npz", allow_pickle=True)['a']
        with open(f'{Params.gel["path2save"]}/embeddings.pkl', 'rb') as handle:
            embeddings = pickle.load(handle)
    except (FileNotFoundError, EOFError) as e:
        cmn.logger.info(f'Loading embeddings failed! Training ...')
        from gel import GraphEmbedding as GE
        embeddings = GE.main(graphs, method=Params.gel['method'])

    # Community Extraction
    cmn.logger.info(f'5. Community Prediction ...')
    cmn.logger.info('#' * 50)
    from cpl import GraphClustering as GC
    try:
        cmn.logger.info(f'Loading user clusters ...')
        Communities = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
    except:
        cmn.logger.info(f'Loading user clusters failed! Generating user clusters ...')
        Communities = GC.main(np.asarray(embeddings), Params.cpl['path2save'], Params.cpl['method'])

    # News Article Recommendation
    cmn.logger.info(f'6. Application: News Recommendation ...')
    cmn.logger.info('#' * 50)
    from apl import News
    news_output = News.main()
    return news_output


def run(tml_baselines, gel_baselines, run_desc):
    for t in tml_baselines:
        for g in gel_baselines:
            try:
                cmn.logger.info(f'Running pipeline for {t} and {g} ....')
                baseline = f'{run_desc}/{t}.{g}'
                with open('ParamsTemplate.py') as f: params_str = f.read()
                new_params_str = params_str.replace('@baseline', baseline).replace('@tml_method', t).replace('@gel_method', g)
                main()
                with open('Params.py', 'w') as f: f.write(new_params_str)
                importlib.reload(Params)
            except:
                cmn.logger.info(traceback.format_exc())
            finally:
                cmn.logger.info('\n\n\n')
    #aggregate('../ouptut')


def aggregate(output_path):
    pred_eval_mean_path = sorted(glob.glob(f'{output_path}/*/apl/evl/Pred.Eval.Mean.csv'))
    pred_eval_mean_agg = pd.DataFrame()
    for i, path in enumerate(pred_eval_mean_path):
        pred_eval_mean = pd.read_csv(path)
        tml_gel = path.split('\\')[-4]
        df = pd.DataFrame(pred_eval_mean.score.values.reshape(1, pred_eval_mean.count()['metric']), index=[tml_gel], columns=pred_eval_mean.metric.values)
        pred_eval_mean_agg = pd.concat((df, pred_eval_mean_agg))
    pred_eval_mean_agg.to_csv(f'{output_path}/Pred.Eval.Mean.Agg.csv')
    return pred_eval_mean_agg

def remove_files():
    try: os.remove('decoder_model_testing.json') # 3 KB
    except: pass
    try: os.remove('decoder_weights_testing.hdf5') # 39 MB
    except: pass
    try: os.remove('encoder_model_testing.json') # 3 KB
    except: pass
    try: os.remove('encoder_weights_testing.hdf5') # 39 MB
    except: pass
    try: os.remove('embedding_testing.txt') # 14 MB
    except: pass
    try: os.remove('next_pred_testing.txt') # 9 GB
    except: pass
    
def addargs(parser):
    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-t', '--tml-method-list', nargs='+', required=True, help='a list of topic modeling methods (eg. -t LDA)')
    baseline.add_argument('-g', '--gel-method-list', nargs='+', required=True, help='a list of graph embedding methods (eg. -g DynAERNN)')
    baseline.add_argument('-r', '--run-desc', required=True, help='a unique description for the run (eg. -r toy')

# python -u main.py -r toy -t LDA -g AE DynAE DynAERNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEERa')
    addargs(parser)
    args = parser.parse_args()
    if not os.path.isdir(f'../output/{args.run_desc}'): os.makedirs(f'../output/{args.run_desc}')
    cmn.logger = cmn.LogFile(f'../output/{args.run_desc}/log.txt')
    run(tml_baselines=args.tml_method_list, gel_baselines=args.gel_method_list, run_desc=args.run_desc)
    aggregate(f'../output/{args.run_desc}')
    remove_files()
