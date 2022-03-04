
# A Framework for Community Prediction in Online Social Networks with Streaming Content
This is an open-source ``extensible`` ``end-to-end`` python-based [``framework``](https://martinfowler.com/bliki/InversionOfControl.html) to predict the future user communities in a text streaming social network (e.g., Twitter) based on the users’ topics of interest. User community prediction aims at identifying communities in the future based on the users' temporal topics of interest. We model inter-user topical affinities at each time interval via streams of temporal graphs. Our framework benefits from temporal graph embedding methods to learn temporal vector representations for users as users' topics of interests and hence their inter-user topical affinities are changing in time. We predict user communities in future time intervals based on the final locations of users' vectors in the latent space. Our framework employs ``layered software design`` that adds modularity, maintainability,  ease of extensibility, and stability against customization and ad hoc changes to its components including ``topic modeling``, ``user modeling``, ``temporal user embedding``, ``user community prediction`` and ``evaluation``. More importantly, our framework further offers one-stop shop access to future communities to improve recommendation systems and advertising campaigns. Our proposed framework has already been benchmarked on a Twitter dataset and showed improvements compared to the state of the art in underlying applications such as ``news article recommendation`` and ``user prediction`` (see [here](https://hosseinfani.github.io/res/papers/Temporal%20Latent%20Space%20Modeling%20For%20Community%20Prediction.pdf), also below).

1. [Demo](#1-Demo)
2. [Structure](#2-Structure)
3. [Setup](#3-Setup)
4. [Quick Start](#4-Quick-Start)
5. [Result](#5-Result)
6. [License](#6-License)
7. [Citation](#7-Citation)

## 1. Demo
:movie_camera: Video Playlist ([tbd](https://github.com/fani-lab/seera/issues/11)): 1) [Introduction](https://youtu.be/6rsaMldfT_U) -> 2) Installation ->3) Extension

Workflow | Layers
:---------------:|:-------------------------:
![](./demo/flow.jpg) | ![](./demo/layers.jpg)

## 2. Structure

### Framework Structure
Our framework has six major layers: Data Access Layer ([``dal``](./src/dal)), Topic Modeling Layer ([``tml``](./src/tml)), User Modeling Layer ([``uml``](./src/uml)), Graph Embedding Layer ([``gel``](./src/gel)), and Community Prediction Layer ([``cpl``](./src/cpl)). The application layer ([``apl``](./src/apl)), is the last layer, as shown in the above figure.

### Code Structure
```
+---output
+---src
|   +---cmn (common functions)
|   |   \---Common.py
|   |
|   +---dal  (data access layer)
|   |   +---DataPreparation.py
|   |   \---DataReader.py
|   |
|   +---tml  (topic modeling layer)
|   |   \---TopicModeling.py
|   |
|   +---uml (user modeling layer)
|   |   +---UsersGraph.py
|   |   \---UserSimilarities.py
|   |
|   +---gel (graph embedding layer)
|   |   +---GraphEmbedding.py
|   |   \---GraphReconstruction.py
|   |
|   +---cpl (community prediction layer)
|   |   \---GraphClustering.py
|   |
|   +---apl (application layer)
|   |   +---NewsTopicExtraction.py
|   |   +---NewsRecommendation.py
|   |   \---ModelEvaluation.py
|   |
|   +---main.py
|   \---params.py
\---requirements.txt
```

## 3. Setup

It is strongly recommended to use Linux OS for installing the packages and executing the framework. To install packages and dependencies, simply use the following commands in your shell:

```bash
git clone https://github.com/fani-lab/seera.git
cd seera
pip install -r requirements.txt
```
This command installs compatible version of the following libraries:

>* dal: ``mysql-connector-python``
>* tml: ``gensim, tagme, nltk, pandas, requests``
>* gel: ``networkx, dynamicgem``
>* others: ``scikit-network, scikit-learn, sklearn, numpy, scipy, matplotlib``

Also, you need to install ``MAchine Learning for LanguagE Toolkit (mallet)`` from its [``git``](https://github.com/mimno/Mallet) or [``website``](http://mallet.cs.umass.edu/index.php), as a requirement in ``tml``. 

## 4. Quick Start

### Data
We crawled and stored Twitter posts (tweets) for 2 consecutive months. The data is available as ``sql`` scripts at [``ds_twitter``](https://drive.google.com/file/d/1UGMPx2xkcNE0e7PbKLpTY61IwvLsKQyN/view?usp=sharing), including, ``Tweets``, ``TweetEntities``, ``TweetUsers``, ``TagmeAnnotations``, ``NewsTables``, and ``GoldenStandard (for news article recommendation)``.

### Run
This framework contains six different layers. Each layer is affected by multiple parameters.
Some of those parameters are fixed in the code via trial and error. However, major parameters such as number of topics can be adjusted by the user.
They can be modified via '*params.py*' file in root folder.\
After modifying '*params.py*', you can run the framework via '*main.py*' with following command:
```bash
cd src
python main.py
```
### Examples
#### **params.py**
```python
import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 1                         

# SQL setting. Should be set for each mysql instance
user = ''
password = ''
host = ''
database = ''


general = {
    'Comment': '', # Any comment to express more information about the configuration.
}

dal = {
    'start': '2010-12-17', # First date of system activity
    'end': '2011-02-17', # Last day of system activity
    'timeInterval': 1, # Time interval (days) for grouping documents
    'lastRowsNumber': 100000, # Number of picked rows of the dataset for the whole process as a sample
    
    # Following parameters is used to generate corpus from our dataset:
    'userModeling': True, # Aggregates all tweets of a user as a document
    'timeModeling': True, # Aggregate all tweets of a specific day as a document
    'preProcessing': False, # Applying some traditional pre-processing methods on corpus
    'TagME': False, # Apply Tagme on the raw dataset. Set it to False if tagme-dataset is used
    'tagme_GCUBE_TOKEN': "--------------" # Tagme GCUBE TOKEN. For more information, visit: [TagmeHelp](https://sobigdata.d4science.org/web/tagme/tagme-help)
}

tml = {
    'num_topics': 25, # Number of topics that should be extracted from our corpus
    'library': 'gensim', # Used library to extract topics from the corpus. Could be 'gensim' or 'mallet'
    'mallet_home': '--------------', # mallet_home path
    'filterExtremes': True, # Filter very common and very rare terms in all documents
    'JO': False, # (JO:=JustOne) If True, just one topic is chosen for each document
    'Bin': True, # (Bin:=Binary) If True, all scores above/below a threshold is set to 1/0 for each topic
    'Threshold': 0.2, # A threshold for topic scores quantization
    'path2saveTML': f'../output/{RunID}/tml'
}

uml = {
    'RunId': RunID, # A unique number to identify the configuration per run
    'UserSimilarityThreshold': 0.2, # A threshold for filtering low user similarity scores
    'path2saveUML': f'../output/{RunID}/uml'
}

gel = {
    'GraphEmbedding': 'Node2Vec', # Graph embedding method. Available options are ['Node2Vec', 'AE', 'DynAE', 'DynRNN', 'DynAERNN']
    'EmbeddingDim': 40, # Embedding dimension
    'path2saveGEL': f'../output/{RunID}/gel'
}

cpl = {
    'ClusteringApproach': 'Indirect', # Available options are ['Direct', 'Indirect']. 'Direct': Applying a non-graph clustering method directly on predicted communities in latent space; 'Indirect': Apply a graph clustering method on generated graph based on the output of predicted communities
    'ClusteringMethod': 'Louvain', # Specification of the clustering method based on 'ClusteringApproach'. The only available option is 'Louvain' ('ClusteringApproach': 'Indirect') which is a graph clustering method
}

evl = {
    'RunId': RunID,
    'EvaluationType': 'Extrinsic', # ['Intrinsic', 'Extrinsic']
    
    # If 'EvaluationType' is set to 'Intrinsic', two below parameters should set as well
    'EvaluationMetrics': ['adjusted_rand', 'completeness', 'homogeneity', 'rand', 'v_measure',
                          'normalized_mutual_info', 'adjusted_mutual_info', 'mutual_info', 'fowlkes_mallows'],
    'GoldenStandardPath': '/path2GS', # Path to the golden standard
    # ----------------------------------------------------------------------------------
}

application = {
    'Threshold': 0.2, # A threshold for filtering low news article recommendation scores
    'TopK': 20 # Number of selected top news article recommendation candidates
}
```
## 5. Result
<table style="color:#828282;">
    <tr style=" color:black;" align="center">
        <th style="background-color:#A8A8A8;" rowspan="2">Method</th>
        <th style="background-color:#A8A8A8;" colspan="3">News Recommendation</th>
        <th style="background-color:#A8A8A8;" colspan="3">User Prediction</th>
    </tr>
    <tr style="color:black;">
        <th style="background-color:#A8A8A8;">mrr</th>
        <th style="background-color:#A8A8A8;">ndcg5</th>
        <th style="background-color:#A8A8A8;">ndcg10</th>
        <th style="background-color:#A8A8A8;">Precision</th>
        <th style="background-color:#A8A8A8;">Recall</th>
        <th style="background-color:#A8A8A8;">f1-measure</th>
    </tr>
    <tr>
        <th style="text-align:left; color:black; background-color:#FBF0CE" colspan="7"> Community Prediction </th>
    </tr>
    <tr>
        <th style="color:black;">Our approach</th>
        <th style="color:black;">0.255</th>
        <th style="color:black;">0.108</th>
        <th style="color:black;">0.105</th>
        <th style="color:black;">0.012</th>
        <th>0.035</th>
        <th style="color:black;">0.015</th>
    </tr>
    <tr>
        <th style="color:black;">Appel et al. [PKDD' 18]</th>
        <th>0.176</th>
        <th>0.056</th>
        <th>0.055</th>
        <th>0.007</th>
        <th>0.094</th>
        <th>0.0105</th>
    </tr>
        <th style="text-align:left; color:black; background-color:#FBF0CE" colspan="7"> Temporal community detection </th>
        <tr>
        <th style="color:black;">Hu et al. [SIGMOD’15]</th>
        <th>0.173</th>
        <th>0.056</th>
        <th>0.049</th>
        <th>0.007</th>
        <th>0.136</th>
        <th>0.013</th>
    </tr>
        <th style="color:black;">Fani et al.    [CIKM’17]</th>
        <th>0.065</th>
        <th>0.040</th>
        <th>0.040</th>
        <th>0.007</th>
        <th>0.136</th>
        <th>0.013</th>
    </tr>
        <th style="text-align:left; color:black; background-color:#FBF0CE" colspan="7"> Non-temporal link-based community detection </th>
    <tr>
        <th style="color:black;">Ye et al.[CIKM’18]</th>
        <th>0.139</th>
        <th>0.056</th>
        <th>0.055</th>
        <th>0.008</th>
        <th>0.208</th>
        <th>0.014</th>
    </tr>
        <th style="color:black;">Louvain[JSTAT’08]</th>
        <th>0.108</th>
        <th>0.048</th>
        <th>0.055</th>
        <th>0.004</th>
        <th>0.129</th>
        <th>0.007</th>
    </tr>
        <th style="text-align:left; color:black;background-color:#FBF0CE" colspan="7"> Collaborative filtering </th>
    </tr>
        <th style="color:black;">rrn[WSDM’17]</th>
        <th>0.173</th>
        <th>0.073</th>
        <th>0.08</th>
        <th>0.004</th>
        <th style="color:black;">0.740</th>
        <th>0.008</th>
    </tr>
    <tr>
        <th style="color:black;">timesvd++    [KDD’08]</th>
        <th>0.141</th>
        <th>0.058</th>
        <th>0.064</th>
        <th>0.003</th>
        <th>0.657</th>
        <th>0.005</th>
    </tr>
</table>

## 6. License
©2021. This work is licensed under a [CC BY-NC-SA 4.0](LICENSE.txt) license.

### Contact
Email: [ziaeines@uwindsor.ca](mailto:ziaeines@uwindsor.ca), [soroushziaeinejad@gmail.com](mailto:soroushziaeinejad@gmail.com)

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### Acknowledgments
In this work, we use [``dynamicgem``](https://github.com/Sujit-O/dynamicgem), [``mallet``](https://github.com/mimno/Mallet), [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval) and other libraries. We would like to thank the authors of these libraries.

## 7. Citation
```
@inproceedings{DBLP:conf/ecir/FaniBD20,
  author    = {Hossein Fani and Ebrahim Bagheri and Weichang Du},
  title     = {Temporal Latent Space Modeling for Community Prediction},
  booktitle = {Advances in Information Retrieval - 42nd European Conference on {IR} Research, {ECIR} 2020, Lisbon, Portugal, April 14-17, 2020, Proceedings, Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12035},
  pages     = {745--759},
  publisher = {Springer},
  year      = {2020},
  url       = {https://doi.org/10.1007/978-3-030-45439-5\_49},
  doi       = {10.1007/978-3-030-45439-5\_49},
  timestamp = {Thu, 14 May 2020 10:17:16 +0200},
  biburl    = {https://dblp.org/rec/conf/ecir/FaniBD20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
