from shutil import copyfile
import os, glob, pickle, argparse, importlib, traceback
import numpy as np
import pandas as pd
import gensim
import torch
import datetime

import params
from cmn import Common as cmn


def main():
    if not os.path.isdir(f'../output/{params.general["baseline"]}'): os.makedirs(
        f'../output/{params.general["baseline"]}')
    copyfile('params.py', f'../output/{params.general["baseline"]}/params.py')

    os.environ["CUDA_VISIBLE_DEVICES"] = params.general['cuda']

    cmn.logger.info(f'1. Data Reading & Preparation ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading perprocessed files ...')
        from ast import literal_eval
        documents = pd.read_csv(f"../output/{params.general['baseline']}/documents.csv", converters={"Extracted_Links":literal_eval, "Tokens": literal_eval})
    except (FileNotFoundError, EOFError) as e:
        from dal import DataReader as dr, DataPreparation as dp
        cmn.logger.info(f'Loading perprocessed files failed! Generating files ...')
        dataset = dr.load_tweets(f'{params.dal["path"]}/Tweets.csv')
        cmn.logger.info(f'dataset.shape: {dataset.shape}')
        cmn.logger.info(f'dataset.keys: {dataset.keys()}')

        cmn.logger.info(f'Data Preparation ...')
        documents = dp.data_preparation(dataset)

    cmn.logger.info(f'documents.shape: {documents.shape}')
    cmn.logger.info(f'2. Topic modeling ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading topic modeling model ...')
        if params.tml['method'].lower() == 'lda':
            dictionary = gensim.corpora.Dictionary.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['numTopics']}topics_TopicModelingDictionary.mm")
            tm_model = gensim.models.LdaModel.load(f"{params.tml['path2save']}/{params.tml['library']}_{params.tml['numTopics']}topics.model")
        elif params.tml['method'].lower() == 'gsdmm':
            dictionary = gensim.corpora.Dictionary.load(f"{params.tml['path2save']}/{params.tml['method']}_{params.tml['numTopics']}topics_TopicModelingDictionary.mm")
            tm_model = pd.read_pickle(f"{params.tml['path2save']}/{params.tml['method']}_{params.tml['numTopics']}topics.model")
        elif params.tml['method'].lower() == 'btm':
            dictionary = pd.read_pickle(f"{params.tml['path2save']}/{params.tml['method']}_{params.tml['numTopics']}topics_TopicModelingDictionary.mm")
            tm_model = pd.read_pickle(f"{params.tml['path2save']}/{params.tml['method']}_{params.tml['numTopics']}topics.model")
        elif params.tml['method'].lower() == 'random':
            dictionary = gensim.corpora.Dictionary.load(f"{params.tml['path2save']}/{params.tml['method']}_{params.tml['numTopics']}topics_TopicModelingDictionary.mm")
            tm_model = None
    except (FileNotFoundError, EOFError) as e:
        from tml import TopicModeling as tm
        cmn.logger.info(f'Loading topic modeling model failed! Training a new model ...')
        dictionary, tm_model = tm.topic_modeling(documents)
    cmn.logger.info(f'dictionary.shape: {len(dictionary)}')

    # User Graphs
    cmn.logger.info(f"3. Temporal Graph Creation ...")
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f"Loading users' graph stream ...")
        try:
            with open(f'{params.uml["path2save"]}/graphs/graphs.pkl', 'rb') as f:
                graphs = pickle.load(f)
                cmn.logger.info(f"Pickle loaded the dataset")
        except:
            graphs = torch.load(f'{params.uml["path2save"]}/graphs/graphs.pt')
            cmn.logger.info(f"Torch loaded the dataset")

    except (FileNotFoundError, EOFError) as e:
        from uml import UserSimilarities as US
        cmn.logger.info(f"Loading users' graph stream failed! Generating the stream ...")
        graphs = US.main(documents, dictionary, tm_model)

    # Graph Embedding
    cmn.logger.info(f'4. Temporal Graph Embedding ...')
    cmn.logger.info('#' * 50)
    try:
        cmn.logger.info(f'Loading embeddings ...')
        from ast import literal_eval
        user_features = pd.read_csv(f'{params.gel["path2save"]}/userFeatures.csv', converters={"FinalInterests":literal_eval})
        predicted_features = torch.load(f'{params.gel["path2save"]}/embeddings.pt')
    except (FileNotFoundError, EOFError) as e:
        cmn.logger.info(f'Loading embeddings failed! Training ...')
        from gel import TorchGraphEmbedding as TGE
        user_features, predicted_features = TGE.main(documents, graphs)

    # Community Extraction
    cmn.logger.info(f'5. Community Prediction ...')
    cmn.logger.info('#' * 50)

    try:
        cmn.logger.info(f'Loading user clusters ...')
        Communities = np.load(f'{params.cpl["path2save"]}/PredUserClusters.npy', allow_pickle=True)
        user_clusters = pd.read_csv(f'{params.cpl["path2save"]}/user_clusters.csv')
    except:
        from cpl import GraphClustering as GC
        cmn.logger.info(f'Loading user clusters failed! Generating user clusters ...')
        user_clusters, Communities = GC.main2(user_features, predicted_features)

    # News Article Recommendation
    cmn.logger.info(f'6. Application: News Recommendation ...')
    cmn.logger.info('#' * 50)
    from apl import News, NewsUsers
    # News Dataset Creation:
    # if params.apl['crawlURLs']:
    #     documents, extracted_info = NewsUsers.main(documents)
    news_output = News.main(user_features, user_clusters, dictionary, tm_model)
    return news_output

def run(tml_baselines, gel_baselines, run_desc):
    for t in tml_baselines:
        for g in gel_baselines:
            try:
                cmn.logger.info(f'Running pipeline for {t} and {g} ....')
                baseline = f'{run_desc}/{t}.{g}'
                with open('params_template.py') as f:
                    params_str = f.read()
                new_params_str = params_str.replace('@baseline', baseline).replace('@tml_method', t).replace(
                    '@gel_method', g)
                with open('params.py', 'w') as f:
                    f.write(new_params_str)
                importlib.reload(params)
                main()
            except:
                cmn.logger.info(traceback.format_exc())
            finally:
                cmn.logger.info('\n\n\n')
    # aggregate('../ouptut')

def aggregate(output_path):
    pred_eval_mean_path = sorted(glob.glob(f'{output_path}/*/apl/evl/pred.eval.mean.csv'))
    pred_eval_mean_agg = pd.DataFrame()
    for i, path in enumerate(pred_eval_mean_path):
        pred_eval_mean = pd.read_csv(path)
        tml_gel = path.split('\\')[-4]
        df = pd.DataFrame(pred_eval_mean.score.values.reshape(1, pred_eval_mean.count()['metric']), index=[tml_gel],
                          columns=pred_eval_mean.metric.values)
        pred_eval_mean_agg = pd.concat((df, pred_eval_mean_agg))
    pred_eval_mean_agg.T.to_csv(f'{output_path}/pred.eval.mean.agg.csv')
    return pred_eval_mean_agg

def remove_files():
    try: os.remove('decoder_model_testing.json')  # 3 KB
    except: pass
    try: os.remove('decoder_weights_testing.hdf5')  # 39 MB
    except: pass
    try: os.remove('encoder_model_testing.json')  # 3 KB
    except: pass
    try: os.remove('encoder_weights_testing.hdf5')  # 39 MB
    except: pass
    try: os.remove('embedding_testing.txt')  # 14 MB
    except: pass
    try: os.remove('next_pred_testing.txt')  # 9 GB
    except: pass

def addargs(parser):
    baseline = parser.add_argument_group('baseline')
    baseline.add_argument('-t', '--tml-method-list', nargs='+', required=True,
                          help='a list of topic modeling methods (eg. -tml_models LDA)')
    baseline.add_argument('-g', '--gel-method-list', nargs='+', required=True,
                          help='a list of graph embedding methods (eg. -gel_models DynAERNN)')
    baseline.add_argument('-r', '--run-desc', required=True, help='a unique description for the run (eg. -run_desc toy')


# python -u main.py -r toy -t LDA -g AE DynAE DynAERNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEERa')
    addargs(parser)
    args = parser.parse_args()
    if not os.path.isdir(f'../output/{args.run_desc}'): os.makedirs(f'../output/{args.run_desc}')
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cmn.logger = cmn.LogFile(f'../output/{args.run_desc}/log_{current_time}.txt')
    run(tml_baselines=args.tml_method_list, gel_baselines=args.gel_method_list, run_desc=args.run_desc)
    aggregate(f'../output/{args.run_desc}')
    # remove_files()
