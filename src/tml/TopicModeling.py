import os, sys
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.coherencemodel import CoherenceModel
#import pyLDAvis
#import pyLDAvis.gensim

#sys.path.extend(["../"])
from cmn import Common as cmn
import params

def topic_modeling(processed_docs, num_topics, filterExtremes, library, path_2_save_tml):
    if not os.path.isdir(path_2_save_tml): os.makedirs(path_2_save_tml)
    cmn.logger.info(f'TopicModeling: num_topics={num_topics},  filterExtremes={filterExtremes}, library={library}')
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if filterExtremes: dictionary.filter_extremes(no_below=1, no_above=0.20, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    if library == 'gensim':
        lda_model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=5)
        lda_model.save(f"{path_2_save_tml}/gensim_{num_topics}topics.model")
        GENSIM_Topics = []
        GENSIM_Percentages = []
        GENSIM = []
        for idx, topic in lda_model.print_topics(-1):
            cmn.logger.info(f'TopicModeling: GENSIM Topic: {idx} \nWords: {topic}')
            splitted = topic.split('+')
            GENSIM.append([])
            GENSIM_Topics.append([])
            GENSIM_Percentages.append([])
            for word in splitted:
                GENSIM_Topics[-1].append(word.split('*')[1].split('"')[1])
                GENSIM_Percentages[-1].append(word.split('*')[0])
                GENSIM[-1].append(word.split('*')[0])
                GENSIM[-1].append(word.split('*')[1].split('"')[1])
        G = []
        for i in range(len(GENSIM_Topics)):
            G.append(GENSIM_Topics[i])
            G.append(GENSIM_Percentages[i])
        np.savetxt(f"{path_2_save_tml}/gensim_{num_topics}topics.csv", G, delimiter=",", fmt='%s')
        totalTopics = GENSIM_Topics
    elif library == 'mallet':
        os.environ['MALLET_HOME'] = params.tml['mallet_home']
        mallet_path = f'{params.tml["mallet_home"]}/bin/mallet'
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=num_topics,
                                                     id2word=dictionary)
        lda_model.save(f"{path_2_save_tml}/mallet_{num_topics}topics.model")
        MALLET = []
        MALLET_Topics = []
        MALLET_Percentages = []
        for idx, topic in lda_model.print_topics(-1):
            cmn.logger.info(f'TopicModeling: MALLET Topic: {idx} \nWords: {topic}')
            splitted = topic.split('+')
            MALLET.append([])
            MALLET_Topics.append([])
            MALLET_Percentages.append([])
            for word in splitted:
                MALLET_Topics[-1].append(word.split('*')[1].split('"')[1])
                MALLET_Percentages[-1].append(word.split('*')[0])
                MALLET[-1].append(word.split('*')[0])
                MALLET[-1].append(word.split('*')[1].split('"')[1])
        M = []
        for i in range(len(MALLET_Topics)):
            M.append(MALLET_Topics[i])
            M.append(MALLET_Percentages[i])
        np.savetxt(f"{path_2_save_tml}/Mallet_{num_topics}topics.csv", M, delimiter=",", fmt='%s', encoding="utf-8")
        totalTopics = MALLET_Topics
    else:
        raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
        pass

    try:
        cmn.logger.info(f'TopicModeling: Coherences:\n')
        c, cv = coherence(dictionary, bow_corpus, totalTopics, lda_model)
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

    return dictionary, bow_corpus, totalTopics, lda_model


def coherence(dictionary, bow_corpus, topics, lda_model):
    cmn.logger.info(f'TopicModeling: Calculating model coherence:\n')
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
    coherenceValue = cm.get_coherence()
    cm = CoherenceModel(topics=topics, dictionary=dictionary, corpus=bow_corpus, coherence='u_mass')
    topicCoherence = cm.get_coherence_per_topic()
    return coherenceValue, topicCoherence


def visualization(dictionary, bow_corpus, lda_model, num_topics, path_2_save_tml='../../output/tml'):
    try:
        visualisation = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        modelStr = 'gensim'
    except:
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
        visualisation = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary)
        modelStr = 'mallet'

    pyLDAvis.save_html(visualisation, f'{path_2_save_tml}/{modelStr}_LDA_Visualization_{num_topics}topics.html')
    print(os.getcwd())
    print(f'saved in {path_2_save_tml}/{modelStr}_LDA_Visualization_{num_topics}topics.html')
    return 'Visualization is finished'

def doc2topics(ldaModel, doc, threshold=0.2, justOne=True, binary=True):
    doc_topic_vector = np.zeros((ldaModel.num_topics))
    try:
        d2tVector = ldaModel.get_document_topics(doc)
    except:
        gen_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldaModel)
        d2tVector = gen_model.get_document_topics(doc)
    a = np.asarray(d2tVector)
    if len(a) == 0:
        return np.zeros(ldaModel.num_topics)
    if justOne:
        doc_topic_vector[a[:, 1].argmax()] = 1
    else:
        # print(a)
        # threshold = a[:, 1].max()
        for i in d2tVector:
            if i[1] >= threshold:
                if binary:
                    doc_topic_vector[i[0]] = 1
                else:
                    doc_topic_vector[i[0]] = i[1]
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

