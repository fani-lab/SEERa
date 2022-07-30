
# ``SEERa``: An Open-Source Framework for Future Community Prediction 
This is an open-source ``extensible`` ``end-to-end`` python-based [``framework``](https://martinfowler.com/bliki/InversionOfControl.html) to predict the future user communities in a text streaming social network (e.g., Twitter) based on the users’ topics of interest. User community prediction aims at identifying communities in the future based on the users' temporal topics of interest. We model inter-user topical affinities at each time interval via streams of temporal graphs. Our framework benefits from temporal graph embedding methods to learn temporal vector representations for users as users' topics of interests and hence their inter-user topical affinities are changing in time. We predict user communities in future time intervals based on the final locations of users' vectors in the latent space. Our framework employs ``layered software design`` that adds modularity, maintainability,  ease of extensibility, and stability against customization and ad hoc changes to its components including ``topic modeling``, ``user modeling``, ``temporal user embedding``, ``user community prediction`` and ``evaluation``. More importantly, our framework further offers one-stop shop access to future communities to improve recommendation systems and advertising campaigns. Our proposed framework has already been benchmarked on a Twitter dataset and showed improvements compared to the state of the art in underlying applications such as ``news article recommendation`` and ``user prediction`` (see [here](https://hosseinfani.github.io/res/papers/Temporal%20Latent%20Space%20Modeling%20For%20Community%20Prediction.pdf), also below).

1. [Demo](#1-Demo)
2. [Structure](#2-Structure)
3. [Setup](#3-Setup)
4. [Quickstart](#4-Quickstart)
5. [Benchmark Result](#5-Benchmark-Result)
6. [License](#6-License)
7. [Citation](#7-Citation)

## 1. :movie_camera: Demo
Tutorials: 1) [Overview](https://youtu.be/ovH-_KHDvFE) 2) [Quickstart](https://youtu.be/EcG6riy_qOA) ([``Colab Notebook``](https://colab.research.google.com/github/fani-lab/SEERa/blob/main/quickstart.ipynb)) 3) Extension

Workflow | Layers
:---------------:|:-------------------------:
![](./demo/flow.jpg) | ![](./demo/layers.jpg)

## 2. Structure

### Framework Structure
Our framework has six major layers: Data Access Layer ([``dal``](./src/dal)), Topic Modeling Layer ([``tml``](./src/tml)), User Modeling Layer ([``uml``](./src/uml)), Graph Embedding Layer ([``gel``](./src/gel)), and Community Prediction Layer ([``cpl``](./src/cpl)). The application layer ([``apl``](./src/apl)), is the last layer, as shown in the above figure.

Each layer process the input data from previous layer and produces new processed data for the next layer as explained below. Sample outputs on [``toy``](./data/toy) data can be seen here [``./output/toy``](./output/toy):

#### [``tml``](./src/tml)
```
gensim[or Mallet]_{#Topics}topics.csv                           -> N topics with their top 10 vocabulary set and probabilities
gensim[or Mallet]_{#Topics}topics.model                         -> The LDA model
gensim[or Mallet]_{#Topics}topics_TopicModelingDictionary.mm    -> LDA dictionary
```
#### [``uml``](./src/uml)
```
Day{K}UserIDs.npy               -> User IDs for K-th day [Size: #Users × 1]
Day{K}UsersTopicInterests.npy   -> Matrix of users to topics [Size: #Users × #Topics]
users.npy                       -> User IDs [Size: #Users × 1]
```
#### [``gel``](./src/gel)
```
embeddings.npz -> embedded user graphs [Size: #Days-loockback × #Users × Embedding dim]
```
#### [``cpl``](./src/cpl)
```
Graph.net[.pkl] -> Final predicted user graph for the future from last embeddings
PredUserClusters.npy[.csv] -> Cluster ID for each user [Size: #Users × 1]
```
#### [``apl``](./src/apl)
```
ClusterNumbers.npy              -> Cluster IDs [Size: #Communities]
NewsIds.npy                     -> News IDs [Size: #News × 1]
CommunitiesTopicInterests.npy   -> Topic vector for each community [Size: #Communities × #Topics]
NewsTopics.npy                  -> Topic vector for each news article [Size: #News × #Topics]
RecommendationTable.npy         -> Recommendations scores of news articles for each community [Size: #Communities × #News]
TopRecommendations.npy          -> TopK recommendations scores of news articles for each community [Size: #Communities × TopK]
```
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
+---environment.yml
\---requirements.txt
```

## 3. Setup

`SEERa` has been developed on `Python 3.6` and can be installed by `conda` or `pip`:

```bash
git clone https://github.com/fani-lab/seera.git
cd seera
conda env create -f environment.yml
conda activate seera
```

```bash
git clone https://github.com/fani-lab/seera.git
cd seera
pip install -r requirements.txt
```

This command installs compatible versions of the following libraries:

>* tml: ``gensim, tagme, nltk, pandas, requests``
>* gel: ``networkx``
>* others: ``scikit-network, scikit-learn, sklearn, numpy, scipy, matplotlib``

Additionally, you need to install the following libraries from their source:
- [``MAchine Learning for LanguagE Toolkit (mallet)``](http://mallet.cs.umass.edu/index.php) as a requirement in ``tml``.
- [``DynamicGem``](https://github.com/palash1992/DynamicGEM) as a requirement in ``gel``:
```bash
git clone https://github.com/palash1992/DynamicGEM.git
cd DynamicGEM
python setup.py install
pip install tensorflow==1.11.0 --force-reinstall #may be needed
```

## 4. Quickstart

### Data
We crawled and stored `~2.9M` Twitter posts (tweets) for 2 consecutive months `2010-11-01` and `2010-12-31`. Tweet Ids are provided at [`./data/TweetIds.csv`](./data/) for streaming tweets from Twitter using tools like [`hydrator`](https://github.com/DocNow/hydrator).

For quickstart purposes, a `toy` sample of tweets between `2010-12-01` and `2010-12-04` has been provided at [`./data/toy/Tweets.csv`](./data/toy).  

### Run
This framework contains six different layers. Each layer is affected by multiple parameters, e.g., number of topics, that can be adjusted by the user via [`./src/params_template.py`](./src/params_template.py) in root folder.

You can run the framework via [`./src/main.py`](./src/main.py) with following command:
```bash
cd src
python -u main.py -run_desc toy -tml_methods LDA -gel_methods AE DynAE DynAERNN
```
where the input arguements are:

`-tml_methods`: A list of topic modeling methods among {`LDA`}, required.

`-gel_methods`: A list of graph embedding methods among {`AE`, `DynAE`, `DynRNN`, `DynAERNN`}, required.

`-run_desc`: A unique description for the run, required.

A run will produce an output folder at `./output/{run_desc}` and subfolders for each topic modeling and graph embedding pair as baselines, e.g., `LDA.AE`, `LDA.DynAE`, and `LDA.DynAERNN`. The final evaluation results are aggregated in `./output/{run_desc}/pred.eval.mean.csv`. See an example run on toy dataset at [`./output/toy`](./output/toy). 

## 5. Benchmark Result
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
        <th style="color:black;"><a href="https://link.springer.com/chapter/10.1007/978-3-030-45439-5_49">Fani et al.[ECIR'20]</a></th>
        <th style="color:black;">0.255</th>
        <th style="color:black;">0.108</th>
        <th style="color:black;">0.105</th>
        <th style="color:black;">0.012</th>
        <th>0.035</th>
        <th style="color:black;">0.015</th>
    </tr>
    <tr>
        <th style="color:black;"><a href="https://link.springer.com/chapter/10.1007/978-3-030-10928-8_1">Appel et al. [PKDD'18]</a></th>
        <th>0.176</th>
        <th>0.056</th>
        <th>0.055</th>
        <th>0.007</th>
        <th>0.094</th>
        <th>0.0105</th>
    </tr>
        <th style="text-align:left; color:black; background-color:#FBF0CE" colspan="7"> Temporal community detection </th>
        <tr>
        <th style="color:black;">Hu et al. [SIGMOD'15]</th>
        <th>0.173</th>
        <th>0.056</th>
        <th>0.049</th>
        <th>0.007</th>
        <th>0.136</th>
        <th>0.013</th>
    </tr>
        <th style="color:black;">Fani et al. [CIKM'17]</th>
        <th>0.065</th>
        <th>0.040</th>
        <th>0.040</th>
        <th>0.007</th>
        <th>0.136</th>
        <th>0.013</th>
    </tr>
        <th style="text-align:left; color:black; background-color:#FBF0CE" colspan="7"> Non-temporal link-based community detection </th>
    <tr>
        <th style="color:black;">Ye et al.[CIKM'18]</th>
        <th>0.139</th>
        <th>0.056</th>
        <th>0.055</th>
        <th>0.008</th>
        <th>0.208</th>
        <th>0.014</th>
    </tr>
        <th style="color:black;">Louvain[JSTAT'08]</th>
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
        <th style="color:black;">timesvd++    [KDD'08]</th>
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
