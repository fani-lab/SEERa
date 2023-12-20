import os, sys
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.coherencemodel import CoherenceModel

from cmn import Common as cmn
import params

def topic_modeling(documents):
    processed_docs = documents['Tokens']
    if not os.path.isdir(params.tml['path2save']): os.makedirs(params.tml['path2save'])
    cmn.logger.info(f'TopicModeling: num_topics={params.tml["numTopics"]},  filterExtremes={params.tml["filterExtremes"]}, library={params.tml["library"]}')
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if params.tml['filterExtremes']: dictionary.filter_extremes(no_below=1, no_above=0.40, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if params.tml['library'] == 'gensim':
        lda_model = gensim.models.LdaModel(bow_corpus, num_topics=params.tml['numTopics'], id2word=dictionary, passes=5)
        lda_model.save(f'{params.tml["path2save"]}/gensim_{params.tml["numTopics"]}topics.model')
        gensim_topics = []
        gensim_percentages = []
        gensim_topics_percentages = []
        for idx, topic in lda_model.print_topics(-1):
            cmn.logger.info(f'TopicModeling: GENSIM Topic: {idx} \nWords: {topic}')
            splitted = topic.split('+')
            gensim_topics.append([])
            gensim_percentages.append([])
            for word in splitted:
                gensim_topics[-1].append(word.split('*')[1].split('"')[1])
                gensim_percentages[-1].append(word.split('*')[0])
        gensim_save = []
        for i in range(len(gensim_topics)):
            gensim_save.append(gensim_topics[i])
            gensim_save.append(gensim_percentages[i])
        np.savetxt(f"{params.tml['path2save']}/gensim_{params.tml['numTopics']}topics.csv", gensim_save, delimiter=",", fmt='%s')
        total_topics = gensim_topics
    elif params.tml['library'] == 'mallet':
        os.environ['MALLET_HOME'] = params.tml['malletHome']
        mallet_path = f'{params.tml["malletHome"]}/bin/mallet'
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=params.tml['numTopics'], id2word=dictionary)
        lda_model.save(f"{params.tml['path2save']}/mallet_{params.tml['numTopics']}topics.model")
        mallet_topics = []
        mallet_percentages = []
        mallet_topics_percentages = []
        for idx, topic in lda_model.print_topics(-1):
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
        np.savetxt(f"{params.tml['path2save']}/Mallet_{params.tml['numTopics']}topics.csv", mallet_save, delimiter=",", fmt='%s', encoding="utf-8")
        total_topics = mallet_topics
    else:
        raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
        pass

    try:
        cmn.logger.info(f'TopicModeling: Coherences:\n')
        c, cv = coherence(dictionary, bow_corpus, total_topics, lda_model)
        cmn.logger.info(f'TopicModeling: Coherence value is: {c}')
        cmn.logger.info(f'TopicModeling: Topic coherences are: {cv}')
    except:
        pass
    dictionary.save(f'{params.tml["path2save"]}/{params.tml["library"]}_{params.tml["numTopics"]}topics_TopicModelingDictionary.mm')
    return dictionary, lda_model


def coherence(dictionary, bow_corpus, topics, lda_model):
    cmn.logger.info(f'TopicModeling: Calculating model coherence:\n')
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
    coherence_value = cm.get_coherence()
    cm = CoherenceModel(topics=topics, dictionary=dictionary, corpus=bow_corpus, coherence='u_mass')
    topic_coherence = cm.get_coherence_per_topic()
    return coherence_value, topic_coherence

def doc2topics(lda_model, doc):
    doc_topic_vector = np.zeros((lda_model.num_topics))
    try:
        d2t_vector = lda_model.get_document_topics(doc)
    except:
        gen_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
        d2t_vector = gen_model.get_document_topics(doc)
    a = np.asarray(d2t_vector)
    if len(a) == 0:
        return np.zeros(lda_model.num_topics)
    if params.tml['justOne']:
        doc_topic_vector[a[:, 1].argmax()] = 1
    else:
        for i in d2t_vector:
            if i[1] >= params.tml['threshold']:
                if params.tml['binary']:
                    doc_topic_vector[i[0]] = 1
                else:
                    doc_topic_vector[i[0]] = i[1]
    doc_topic_vector = np.asarray(doc_topic_vector)
    return doc_topic_vector
