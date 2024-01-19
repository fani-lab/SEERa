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
    if params.tml['filterExtremes']: dictionary.filter_extremes(no_below=1, no_above=0.35, keep_n=200000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if params.tml['method'].lower() == "lda":
        if params.tml['library'].lower() == 'gensim':
            tm_model = gensim.models.LdaModel(bow_corpus, num_topics=params.tml['numTopics'], id2word=dictionary, alpha='auto', eval_every=5, passes=100, iterations=50)
            tm_model.save(f'{params.tml["path2save"]}/gensim_{params.tml["numTopics"]}topics.model')
        elif params.tml['library'] == 'mallet':
            os.environ['MALLET_HOME'] = params.tml['malletHome']
            mallet_path = f'{params.tml["malletHome"]}/bin/mallet'
            tm_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=params.tml['numTopics'], id2word=dictionary)
            tm_model.save(f"{params.tml['path2save']}/mallet_{params.tml['numTopics']}topics.model")
        else:
            raise ValueError("Wrong library name. select 'gensim' or 'mallet'")
            pass
        to_csv(tm_model)
    elif params.tml['method'].lower() == "gsdmm":
        from gsdmm import MovieGroupProcess
        tm_model = MovieGroupProcess(K=params.tml['numTopics'], alpha=0.1, beta=0.1, n_iters=50)
        tm_model.fit(bow_corpus, len(dictionary))
        pd.to_pickle(tm_model, f'{params.tml["path2save"]}/gsdmm_{params.tml["numTopics"]}topics.model')
        to_csv(tm_model, dictionary)
    elif params.tml['method'].lower() == "random":
        tm_model = None
    # Model Evaluation
    if params.tml['eval']:
        cmn.logger.info(f'TopicModeling: Evaluation:\n')
        coh, per = evaluate_lda_model(tm_model, bow_corpus, dictionary)
        cmn.logger.info(f'TopicModeling: model coherence is: {coh}')
        cmn.logger.info(f'TopicModeling: model perplexity is: {per}')
    if params.tml['visualization']:
        cmn.logger.info(f'TopicModeling: Visualization:\n')
        visualization(tm_model, bow_corpus, dictionary)
    if params.tml["method"] == 'lda':
        dictionary.save(f'{params.tml["path2save"]}/{params.tml["library"]}_{params.tml["numTopics"]}topics_TopicModelingDictionary.mm')
    else:
        dictionary.save(f'{params.tml["path2save"]}/{params.tml["method"]}_{params.tml["numTopics"]}topics_TopicModelingDictionary.mm')
    return dictionary, tm_model

def doc2topics(tm_model, doc, dic=None):
    method = params.tml['method'].lower()
    if method == "btm":
        doc = btm.get_vectorized_docs([' '.join(doc)], dic)
        d2t_vector = tm_model.transform(doc)[0]
        # topics_num = tm_model.topics_num_
    elif method == "gsdmm":
        d2t_vector = tm_model.score(doc)
        doc_topic_vector = np.round(np.asarray(d2t_vector), decimals=3)
    elif method == 'lda':
        d2t_vector = tm_model.get_document_topics(doc) if params.tml['library'] == 'gensim' else gensim.models.wrappers.ldamallet.malletmodel2ldamodel(tm_model).get_document_topics(doc)
        topics_num = tm_model.num_topics
        doc_topic_vector = np.zeros((topics_num))
        for index, value in d2t_vector:
            doc_topic_vector[index] = value
        doc_topic_vector = np.round(doc_topic_vector, decimals=3)
    elif method == 'random':
        doc_topic_vector = np.round(np.random.random((params.tml['numTopics'])), decimals=3)
    else:
        raise ValueError("Invalid topic modeling!")
    if params.tml['justOne']: doc_topic_vector[d2t_vector[:, 1].argmax()] = 1
    if params.tml['binary']: doc_topic_vector = np.asarray([1 if val >= params.tml['threshold'] else 0 for val in doc_topic_vector])
    return doc_topic_vector

def evaluate_lda_model(lda_model, corpus, dictionary):
    coherence_model = CoherenceModel(model=lda_model, texts=corpus, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    perplexity = lda_model.log_perplexity(corpus)
    return coherence_score, perplexity

def visualization(lda_model, corpus, dictionary):
    import pyLDAvis.gensim
    import pyLDAvis
    import matplotlib.pyplot as plt
    pyLDAvis.enable_notebook()
    vis_data = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, f'{params.tml["path2save"]}/topic_modeling_vis.html')
    plt.figure(figsize=(10, 7))
    pyLDAvis.display(vis_data)
    plt.show()


# Topics to CSV format
def to_csv(model, dictionary=None):
    if params.tml['method'].lower() == 'lda':
        topics_info = model.show_topics(num_topics=model.num_topics, num_words=10, formatted=False)
        topics = pd.DataFrame()
        for topic_num, words_info in topics_info:
            top_words = [word for word, prob in words_info]
            top_probabilities = [prob for word, prob in words_info]
            topics[f'Topic_{topic_num}_Words'] = top_words
            topics[f'Topic_{topic_num}_Probabilities'] = top_probabilities
        topics.to_csv(f'{params.tml["path2save"]}/final_topics.csv', index=False)
    elif params.tml['method'].lower() == 'gsdmm':
        from collections import Counter
        topic_word_frequencies = {topic: Counter() for topic in range(len(model.cluster_word_distribution))}
        # Count word frequencies for each topic
        for topic, word_ids in enumerate(model.cluster_word_distribution):
            topic_word_frequencies[topic].update(word_ids)
        # Get the top 50 words for each topic based on frequency
        top_words_per_topic = {f'Topic_{topic + 1}': [dictionary[word_id[0]] for word_id, _ in freq.most_common(50)]
                               for topic, freq in topic_word_frequencies.items()}
        # Create a DataFrame to store the results
        topics = pd.DataFrame.from_dict(top_words_per_topic, orient='columns')
        # Save the DataFrame to CSV
        topics.to_csv(f'{params.tml["path2save"]}/final_topics.csv', index=False)