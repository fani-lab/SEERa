import pandas as pd
import numpy as np
import gensim


def doc2topics(lda_model, doc):
    d2t_vector = lda_model.get_document_topics(doc)
    doc_topic_vector = np.zeros(lda_model.num_topics)
    if not d2t_vector: return doc_topic_vector
    for topic, prob in d2t_vector:
        doc_topic_vector[topic] = prob
    return doc_topic_vector


docs = pd.read_csv('uml/documents.csv')
lda_model = gensim.models.LdaModel.load('tml/gensim_30topics.model')
dic = gensim.corpora.Dictionary.load('tml/gensim_30topics_TopicModelingDictionary.mm')
test = 'wikileaks assange julian women house der aids mundo day omg'
tokens = test.split()
res = doc2topics(lda_model, dic.doc2bow(tokens))
print(test)
print(res)
print('----------------------------------------------------')
topics_info = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=100, formatted=False)
for index, element in topics_info:
    if res[index] > 0.1:
        print('***** SELECTED *****')
    top_words = [word for word, prob in element]
    for tok in tokens:
        if tok in top_words:
            print(tok)
    print(top_words)
    print('----------------------------------------------------')
