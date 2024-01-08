import os
import numpy as np
import pandas as pd
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
    elif params.tml['library'] == 'mallet':
        os.environ['MALLET_HOME'] = params.tml['malletHome']
        mallet_path = f'{params.tml["malletHome"]}/bin/mallet'
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=params.tml['numTopics'], id2word=dictionary)
        lda_model.save(f"{params.tml['path2save']}/mallet_{params.tml['numTopics']}topics.model")
    else:
        raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
        pass
    # Topics to CSV format
    topics_info = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=10, formatted=False)
    topics = pd.DataFrame()
    for topic_num, words_info in topics_info:
        top_words = [word for word, prob in words_info]
        top_probabilities = [prob for word, prob in words_info]
        topics[f'Topic_{topic_num}_Words'] = top_words
        topics[f'Topic_{topic_num}_Probabilities'] = top_probabilities
    topics.to_csv(f'{params.tml["path2save"]}/final_topics.csv', index=False)
    try:
        cmn.logger.info(f'TopicModeling: Coherences:\n')
        total_topics = []
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
    if params.tml['library'] == 'gensim': d2t_vector = lda_model.get_document_topics(doc)
    elif params.tml['library'] == 'mallet': d2t_vector = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model).get_document_topics(doc)
    else: raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
    doc_topic_vector = np.zeros(lda_model.num_topics)
    if not d2t_vector: return doc_topic_vector
    if params.tml['justOne']: doc_topic_vector[max(d2t_vector, key=lambda x: x[1])[0]] = 1
    else:
        for topic, prob in d2t_vector:
            doc_topic_vector[topic] = 1 if (prob >= params.tml['threshold'] and params.tml['binary']) else prob
    return doc_topic_vector