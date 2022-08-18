from shutil import copyfile
import sys, os, glob, pickle, argparse, importlib, traceback
from time import time

import numpy as np
import pandas as pd
import gensim
import networkx as nx

import Params
from cmn import Common as cmn


def main():
    if not os.path.isdir(f'../output/{Params.general["baseline"]}'): os.makedirs(f'../output/{Params.general["baseline"]}')
    copyfile('Params.py', f'../output/{Params.general["baseline"]}/Params.py')

    os.environ["CUDA_VISIBLE_DEVICES"] = Params.general['cuda']

    cmn.logger.info(f'\n1. DAL: Temporal Document Creation from Social Posts ...')
    cmn.logger.info('#' * 50)
    try:
        t_s = time()
        configstr = ''
        if Params.dal["userModeling"] and Params.dal["timeModeling"]: configstr = f'\n(User, Time) a document is concat of user\'s posts in each {Params.dal["timeInterval"]} day(s)'
        elif Params.dal["userModeling"]: configstr = f'\n(User) a document is concat of user\'s posts'
        elif Params.dal["timeModeling"]: configstr = f'\n(Time) a document is concat of all posts in each {Params.dal["timeInterval"]} day(s)'
        else:configstr = '\n(Default) a document is a post'
        if Params.dal['tagMe']: '\n(TagMe) elements are TagMe (Wikipedia) concepts'

        path = f"../output/{Params.general['baseline']}/Documents.csv"
        cmn.logger.info(f'1.1. Loading saved temporal documents from  {path} in which {configstr}...')
        with open(path, 'rb') as infile: documents = pd.read_csv(infile, parse_dates=['CreationDate'])
        n_users = len(documents['UserId'].unique()) if 'UserId' in documents.columns else 'N/A'
        n_timeintervals = len(documents['CreationDate'].unique())
        processed_docs = np.load(f"../output/{Params.general['baseline']}/Prosdocs.npz", allow_pickle=True)['a']
    except (FileNotFoundError, EOFError) as e:
        from dal import DataReader as dr, DataPreparation as dp
        cmn.logger.info(f'1.1. Loading temporal documents failed! Creating temporal documents ...')
        cmn.logger.info(f'1.2. Loading social posts ...')
        posts = dr.load_posts(f'{Params.dal["path"]}/Tweets.csv', Params.dal['start'], Params.dal['end'])
        cmn.logger.info(f'(#Posts): ({len(posts)})')
        cmn.logger.info(f'1.3. Creating temporal documents in which {configstr}')
        processed_docs, documents, n_users, n_timeintervals = dp.data_preparation(posts,
                                                        userModeling=Params.dal['userModeling'],
                                                        timeModeling=Params.dal['timeModeling'],
                                                        TagME=Params.dal['tagMe'],
                                                        startDate=Params.dal['start'],
                                                        timeInterval=Params.dal['timeInterval'],
                                                        stopwords=['www', 'RT', 'com', 'http'])

    cmn.logger.info(f'(#ProcessedDocuments, #Documents, #Users, #TimeIntervals): ({len(processed_docs)},{len(documents)},{n_users},{n_timeintervals})')
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')

    cmn.logger.info(f'\n2. TML: Topic Modeling ...')
    cmn.logger.info('#' * 50)
    try:
        t_s = time()
        path_dict = f"{Params.tml['path2save']}/{Params.tml['numTopics']}TopicsDictionary.mm"
        path_mdl = f"{Params.tml['path2save']}/{Params.tml['numTopics']}Topics.model"
        cmn.logger.info(f'2.1. Loading saved topic model of {Params.tml["method"]} from {path_dict} and {path_mdl} ...')
        dictionary = gensim.corpora.Dictionary.load(path_dict)
        lda_model = gensim.models.LdaModel.load(path_mdl)
    except (FileNotFoundError, EOFError) as e:
        from tml import TopicModeling as tm
        cmn.logger.info(f'2.1. Loading saved topic model failed! Training a model ...')
        cmn.logger.info(f'(#Topics, Model): ({Params.tml["numTopics"]}, {Params.tml["method"]})')

        dictionary, _, _, lda_model, c, cv = tm.topic_modeling(processed_docs,
                                                        method=Params.tml['method'],
                                                        num_topics=Params.tml['numTopics'],
                                                        filter_extremes=Params.tml['filterExtremes'],
                                                        path_2_save_tml=Params.tml['path2save'])

        cmn.logger.info(f'2.2. Quality of topics ...')
        cmn.logger.info(f'(MeanCoherence): ({c})')
        cmn.logger.info(f'(#Topic, Topic Coherences): ({Params.tml["numTopics"]}, {cv})')
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')

    # User Graphs
    cmn.logger.info(f"\n3. UML: Temporal Graph Creation ...")
    cmn.logger.info('#' * 50)
    try:
        t_s = time()
        path = f'{Params.uml["path2save"]}/graphs/graphs.pkl'
        cmn.logger.info(f"3.1. Loading users' graph stream from {path} ...")
        with open(path, 'rb') as g: graphs = pickle.load(g)
    except (FileNotFoundError, EOFError) as e:
        from uml import UserSimilarities as US
        cmn.logger.info(f"3.1. Loading users' graph stream failed! Generating the graph stream ...")

        US.main(documents, dictionary, lda_model,
                num_topics=Params.tml['numTopics'],
                path2_save_uml=Params.uml['path2save'],
                just_one=Params.tml['justOne'], binary=Params.tml['binary'], threshold=Params.tml['threshold'])

        graphs_path = glob.glob(f'{Params.uml["path2save"]}/graphs/*.net')
        graphs = []
        for gp in graphs_path: graphs.append(nx.read_gpickle(gp))
        with open(f'{Params.uml["path2save"]}/graphs/graphs.pkl', 'wb') as g: pickle.dump(graphs, g)
    cmn.logger.info(f'(#Graphs): ({len(graphs)})')
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')

    # Graph Embedding
    cmn.logger.info(f'\n4. GEL: Temporal Graph Embedding ...')
    cmn.logger.info('#' * 50)
    try:
        t_s = time()
        cmn.logger.info(f'4.1. Loading embeddings ...')
        with open(f'{Params.gel["path2save"]}/Embeddings.pkl', 'rb') as handle: embeddings = pickle.load(handle)
    except (FileNotFoundError, EOFError) as e:
        cmn.logger.info(f'4.1. Loading embeddings failed! Training {Params.gel["method"]} ...')
        from gel import GraphEmbedding as GE
        embeddings = GE.main(graphs, method=Params.gel['method'])
    cmn.logger.info(f'(#Embeddings, #Dimension) : ({embeddings[0].shape})')
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')

    # Community Extraction
    cmn.logger.info(f'\n5. Community Prediction ...')
    cmn.logger.info('#' * 50)
    try:
        t_s = time()
        cmn.logger.info(f'5.1. Loading future user communities ...')
        communities = np.load(f'{Params.cpl["path2save"]}/PredUserClusters.npy')
    except:
        cmn.logger.info(f'Loading future user communities failed! Predicting future user communities ...')
        from cpl import GraphClustering as GC
        communities = GC.main(np.asarray(embeddings), Params.cpl['path2save'], Params.cpl['method'])
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')

    # News Article Recommendation
    cmn.logger.info(f'\n6. Application: News Recommendation ...')
    cmn.logger.info('#' * 50)
    t_s = time()
    from apl import News
    news_output = News.main()
    return news_output
    cmn.logger.info(f'Time Elapsed: {(time() - t_s)}')


def run(tml_baselines, gel_baselines, run_desc):
    for t in tml_baselines:
        for g in gel_baselines:
            try:
                cmn.logger.info(f'Running pipeline for {t} and {g} ....')
                baseline = f'{run_desc}/{t}.{g}'
                with open('ParamsTemplate.py') as f: params_str = f.read()
                new_params_str = params_str.replace('@baseline', baseline).replace('@tml_method', t).replace('@gel_method', g)
                with open('Params.py', 'w') as f: f.write(new_params_str)
                importlib.reload(Params)
                main()
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
        try: tml_gel = path.split('\\')[-4]
        except: tml_gel = path.split('/')[-4]
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
    baseline.add_argument('-t', '--tml-method-list', nargs='+', type=str.lower, required=True, help='a list of topic modeling methods (eg. -t LDA)')
    baseline.add_argument('-g', '--gel-method-list', nargs='+', type=str.lower, required=True, help='a list of graph embedding methods (eg. -g DynAERNN)')
    baseline.add_argument('-r', '--run-desc', type=str.lower, required=True, help='a unique description for the run (eg. -r toy')

# python -u main.py -r toy -t LdA.GeNsim -g Ae DynAe DynaERnN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEERa')
    addargs(parser)
    args = parser.parse_args()
    if not os.path.isdir(f'../output/{args.run_desc}'): os.makedirs(f'../output/{args.run_desc}')
    cmn.logger = cmn.LogFile(f'../output/{args.run_desc}/Log.txt')
    run(tml_baselines=args.tml_method_list, gel_baselines=args.gel_method_list, run_desc=args.run_desc)
    aggregate(f'../output/{args.run_desc}')
    remove_files()
