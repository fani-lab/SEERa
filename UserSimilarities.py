import gensim
import networkx as nx
import numpy as np
import TopicModeling as TM

TopicModel = TM.topic_modeling(num_topics=30, filterExtremes=True, library='mallet')
dictionary, bow_corpus, totalTopics, lda_model, num_topics, processed_docs, documents = TopicModel




print('Document to topics transformation:')
# There are two options here. One is aggregating dataset again with whole tweets of a user and give it to the Doc2Topic. Two is apply some functions
# like max or mean on the percentage of interests.

## METHOD 1:
dataGroupbyUsers = documents.groupby(['userId'])
documents = dataGroupbyUsers['Text'].apply(lambda x: '\n'.join(x)).reset_index()
bow_corpus = [dictionary.doc2bow(doc) for doc in documents]
D2T = Doc2Topic(lda_model, bow_corpus)
np.save('Docs2Topics.npy', D2T)

## METHOD 2:
D2T = Doc2Topic(lda_model, bow_corpus)
userset = np.asarray(documents['userId'])
userset_distinct = np.sort(np.asarray(list(set(userset))))
for userId in userset_distinct:
    indices = np.where(userset == userId)[0]
    ThisUserMatrix = D2T[indices.min():indices.max]
    # Aggregation Method: Mean, Max, Whatever...




num_users = len(documents['userId'])
user_user_matrix = np.zeros((num_users, num_users))
def Doc2Topic(ldaModel, docs):
    doc_topic = []
    for doc in docs:
        doc_topic_vector = np.zeros((ldaModel.num_topics))
        d2tVector = ldaModel.get_document_topics(doc)
        for i in d2tVector:
            doc_topic_vector[i[0]] = i[1]
        doc_topic.append(doc_topic_vector)
    doc_topic = np.asarray(doc_topic)
    return doc_topic