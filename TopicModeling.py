import gensim
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import os
from DataPreparation import data_preparation as dp


def topic_modeling(num_topics=20, filterExtremes=False, library='gensim'):
    processed_docs = dp(userModeling=True, preProcessing=False, TagME=True)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if filterExtremes:
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    if library == 'gensim':
        lda_model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2)
        GENSIM_Topics = []
        GENSIM_Percentages = []
        GENSIM = []
        for idx, topic in lda_model.print_topics(-1):
            print('\nGENSIM Topic: {} \nWords: {}'.format(idx, topic))
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
        np.savetxt("Gensim_" + str(num_topics) + "topics.csv", G, delimiter=",", fmt='%s')
        totalTopics = GENSIM_Topics
    elif library == 'mallet':
        os.environ['MALLET_HOME'] = 'C:/Users/sorou/mallet-2.0.8'
        mallet_path = 'C:/Users/sorou/mallet-2.0.8/bin/mallet'  # update this path
        lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=num_topics,
                                                     id2word=dictionary)
        MALLET = []
        MALLET_Topics = []
        MALLET_Percentages = []
        for idx, topic in lda_model.print_topics(-1):
            print('\nMALLET Topic: {} \nWords: {}'.format(idx, topic))
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
        np.savetxt("Mallet_" + str(num_topics) + "topics.csv", M, delimiter=",", fmt='%s', encoding="utf-8")
        totalTopics = MALLET_Topics
    else:
        raise ValueError("Wrong library name. select 'gensim' or 'mallet'")

    try:
        print('Coherences:\n')
        c, cv = coherence(dictionary, bow_corpus, totalTopics, lda_model)
        print('Coherence value is:', c)
        print('Topic coherences are:', cv)
    except:
        pass

    try:
        print('Visualization:\n')
        visualization(dictionary, bow_corpus, lda_model, num_topics)
    except:
        pass

    return dictionary, bow_corpus, totalTopics, lda_model, num_topics


def coherence(dictionary, bow_corpus, topics, lda_model):
    print('Calculating model coherence:\n')
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
    coherenceValue = cm.get_coherence()
    cm = CoherenceModel(topics=topics, dictionary=dictionary, corpus=bow_corpus, coherence='u_mass')
    topicCoherence = cm.get_coherence_per_topic()
    return coherenceValue, topicCoherence


def visualization(dictionary, bow_corpus, lda_model, num_topics):
    try:
        visualisation = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        modelStr = 'gensim'
    except:
        model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
        visualisation = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary)
        modelStr = 'mallet'
    pyLDAvis.save_html(visualisation, modelStr + '_LDA_Visualization_' + str(num_topics) + 'topic.html')
    return 'Visualization is finished'
