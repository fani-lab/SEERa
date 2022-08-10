import os, sys
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.coherencemodel import CoherenceModel
from gsdmm import MovieGroupProcess
#import pyLDAvis
#import pyLDAvis.gensim

from cmn import Common as cmn
import Params


def topic_modeling(processed_docs, method, num_topics, filter_extremes, library, path_2_save_tml):
    if not os.path.isdir(path_2_save_tml): os.makedirs(path_2_save_tml)
    cmn.logger.info(f'TopicModeling: num_topics={num_topics},  filterExtremes={filter_extremes}, library={library}')
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if filter_extremes: dictionary.filter_extremes(no_below=1, no_above=0.20, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if method == "gsdmm":
        tm_model = MovieGroupProcess(K=3, alpha=0.1, beta=0.1, n_iters=30)
        output = tm_model.fit(bow_corpus, len(dictionary))
        total_topics = tm_model.cluster_word_distribution
        with open(f"{path_2_save_tml}/Gensim_{tm_model.K}Topics.pkl", 'wb') as g: pickle.dump(tm_model, g)
    if method == "LDA":
        if library == 'gensim':
            tm_model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=5)
            tm_model.save(f"{path_2_save_tml}/Gensim_{num_topics}Topics.model")
            gensim_topics = []
            gensim_percentages = []
            gensim_topics_percentages = []
            for idx, topic in tm_model.print_topics(-1):
                cmn.logger.info(f'TopicModeling: GENSIM Topic: {idx} \nWords: {topic}')
                splitted = topic.split('+')
                gensim_topics_percentages.append([])
                gensim_topics.append([])
                gensim_percentages.append([])
                for word in splitted:
                    gensim_topics[-1].append(word.split('*')[1].split('"')[1])
                    gensim_percentages[-1].append(word.split('*')[0])
                    gensim_topics_percentages[-1].append(word.split('*')[0])
                    gensim_topics_percentages[-1].append(word.split('*')[1].split('"')[1])
            gensim_save = []
            for i in range(len(gensim_topics)):
                gensim_save.append(gensim_topics[i])
                gensim_save.append(gensim_percentages[i])
            np.savetxt(f"{path_2_save_tml}/Gensim{num_topics}Topics.csv", gensim_save, delimiter=",", fmt='%s')
            total_topics = gensim_topics
        elif library == 'mallet':
            os.environ['MALLET_HOME'] = Params.tml['malletHome']
            mallet_path = f'{Params.tml["malletHome"]}/bin/mallet'
            tm_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=num_topics,
                                                         id2word=dictionary)
            tm_model.save(f"{path_2_save_tml}/Mallet{num_topics}Topics.model")
            mallet_topics = []
            mallet_percentages = []
            mallet_topics_percentages = []
            for idx, topic in tm_model.print_topics(-1):
                cmn.logger.info(f'TopicModeling: MALLET Topic: {idx} \nWords: {topic}')
                splitted = topic.split('+')
                mallet_topics_percentages.append([])
                mallet_topics.append([])
                mallet_percentages.append([])
                for word in splitted:
                    mallet_topics[-1].append(word.split('*')[1].split('"')[1])
                    mallet_percentages[-1].append(word.split('*')[0])
                    mallet_topics_percentages[-1].append(word.split('*')[0])
                    mallet_topics_percentages[-1].append(word.split('*')[1].split('"')[1])
            mallet_save = []
            for i in range(len(mallet_topics)):
                mallet_save.append(mallet_topics[i])
                mallet_save.append(mallet_percentages[i])
            np.savetxt(f"{path_2_save_tml}/Mallet{num_topics}Topics.csv", mallet_save, delimiter=",", fmt='%s', encoding="utf-8")
            total_topics = mallet_topics
        else:
            raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
            pass

    try:
        cmn.logger.info(f'TopicModeling: Coherences:\n')
        c, cv = coherence(dictionary, bow_corpus, total_topics, tm_model)
        cmn.logger.info(f'TopicModeling: Coherence value is: {c}')
        cmn.logger.info(f'TopicModeling: Topic coherences are: {cv}')
    except:
        pass

    # try:
    #     print('Visualization:\n')
    #     visualization(dictionary, bow_corpus, lda_model, num_topics)
    #     # logger.critical('Topics are visualized\n')
    # except:
    #     pass

    dictionary.save(f'{path_2_save_tml}/{library}_{num_topics}topics_TopicModelingDictionary.mm')

    return dictionary, bow_corpus, total_topics, tm_model


def coherence(dictionary, bow_corpus, topics, lda_model):
    cmn.logger.info(f'TopicModeling: Calculating model coherence:\n')
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
    coherence_value = cm.get_coherence()
    cm = CoherenceModel(topics=topics, dictionary=dictionary, corpus=bow_corpus, coherence='u_mass')
    topic_coherence = cm.get_coherence_per_topic()
    return coherence_value, topic_coherence


def visualization(dictionary, bow_corpus, lda_model, num_topics, path_2_save_tml=Params.tml['path2save']):
    try:
        visualisation = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        model_name = 'gensim'
    except:
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
        visualisation = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary)
        model_name = 'mallet'

    pyLDAvis.save_html(visualisation, f'{path_2_save_tml}/{model_name}_LDA_Visualization_{num_topics}topics.html')
    cmn.logger.info(f'saved in {path_2_save_tml}/{model_name}_LDA_Visualization_{num_topics}topics.html')
    #print(f'saved in {path_2_save_tml}/{model_name}_LDA_Visualization_{num_topics}topics.html')
    return 'Visualization is finished'


def doc2topics(lda_model, doc, threshold=0.2, just_one=True, binary=True):
    if Params.tml['method'] == "gsdmm":
        doc_topic_vector = np.zeros((lda_model.K))
        d2t_vector = lda_model.score(doc)
    elif Params.tml['method'] == "LDA":
        doc_topic_vector = np.zeros((lda_model.num_topics))
        try:
            d2t_vector = lda_model.get_document_topics(doc)
        except:
            gen_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
            d2t_vector = gen_model.get_document_topics(doc)
    else:
        raise ValueError
    d2t_vector = np.asarray(d2t_vector)
    if len(d2t_vector) == 0:
        return np.zeros(lda_model.num_topics)
    if just_one:
        doc_topic_vector[d2t_vector[:, 1].argmax()] = 1
    else:
        for idx, t in enumerate(d2t_vector):
            if Params.tml['method'] == "gsdmm": t_temp = t
            elif Params.tml['method'] == "LDA": t_temp = t[1]
            if t_temp >= threshold:
                if binary:
                    doc_topic_vector[idx] = 1
                else:
                    doc_topic_vector[idx] = t_temp
    doc_topic_vector = np.asarray(doc_topic_vector)
    return doc_topic_vector

## test
# sys.path.extend(["../"])
# from dal import DataReader as dr
# from dal import DataPreparation as dp
# dataset = dr.load_tweets(Tagme=True, start='2010-12-20', end='2011-01-01', stopwords=['www', 'RT', 'com', 'http'])
# processed_docs, documents = dp.data_preparation(dataset, userModeling=True, timeModeling=True,  preProcessing=False, TagME=False, lastRowsNumber=-1000)
# TopicModel = topic_modeling(processed_docs, num_topics=20, filterExtremes=True, library='mallet', path_2_save_tml='../../output/tml')
#

