
# Community Prediction in Microblogging Social Networks

This is an open-source python-based framework to predict the future user communities in a text streaming social network (e.g., Twitter) based on the users’ topics of interest.

## Installation

It is strongly recommended to use Linux OS for installing the packages and executing the framework. To install packages and dependencies, simply use this command in your shell:

```bash
 pip install -r requirements.txt
```

## Structure

### Framework Structure
Our framework has six major layers: Data Access Layer (DAL),
Topic Modeling Layer (TML), User Modeling Layer (UML), Graph
Embedding Layer (GEL), and Community Prediction Layer (CPL).
The application layer, which is described in Section 3, is the last
layer to show how our method improves the performance of an application. [[1]](#1)

![image info](fig1.png "T")


### Code Structure
│── **output**\
│── **src**\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **cmn** (common functions)\
│&nbsp;&nbsp;&nbsp;│──── *Common.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **dal**  (data access layer)\
│&nbsp;&nbsp;&nbsp;│──── *DataPreparation.py*\
│&nbsp;&nbsp;&nbsp;│──── *DataReader.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **tml**  (topic modeling layer)\
│&nbsp;&nbsp;&nbsp;│──── *TopicModeling.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **uml** (user modeling layer)\
│&nbsp;&nbsp;&nbsp;│──── *UsersGraph.py*\
│&nbsp;&nbsp;&nbsp;│──── *UserSimilarities.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **gel** (graph embedding layer)\
│&nbsp;&nbsp;&nbsp;│──── *GraphEmbedding.py*\
│&nbsp;&nbsp;&nbsp;│──── *GraphReconstruction.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **cpl** (community prediction layer)\
│&nbsp;&nbsp;&nbsp;│──── *GraphClustering.py*\
│&nbsp;&nbsp;&nbsp;│\
│&nbsp;&nbsp;&nbsp;│── **application layer**\
│&nbsp;&nbsp;&nbsp;│──── *NewsTopicExtraction.py*\
│&nbsp;&nbsp;&nbsp;│──── *NewsRecommendation.py*\
│&nbsp;&nbsp;&nbsp;│──── *ModelEvaluation.py*\
│&nbsp;&nbsp;&nbsp;│── *main.py*\
│&nbsp;&nbsp;&nbsp;│── *params.py*\
│── *requirements.txt*


## Usage
This framework contains six different layers. Each layer is affected by multiple parameters.
Some of those parameters are fixed in the code via trial and error. However, major parameters such as number of topics can be adjusted by the user.
They can be modified via '*params.py*' file in root folder.\
After modifying '*params.py*', you can run the framework via '*main.py*' with following command:
```bash
cd src
python main.py
```
## Examples
### **params.py**
```python
import random
import numpy as np

random.seed(0)
np.random.seed(0)
RunID = 59


# SQL setting
# mallet home path
#
uml = {
    'Comment': 'Corrected - Real test',
    'RunId': RunID,

    'start': '2010-12-17',
    'end': '2010-12-17',
    'lastRowsNumber': 100000,

    'num_topics': 25,
    'library': 'gensim',

    'mallet_home': 'C:/Users/sorou/mallet-2.0.8',

    'userModeling': True,
    'timeModeling': True,
    'preProcessing': False,
    'TagME': False,
     

    'filterExtremes': True,
    'JO': False,
    'Bin': True,
    'Threshold': 0.2,
    'UserSimilarityThreshold': 0.2
}

evl = {
    'RunId': RunID,
    'Threshold': 0,
    'TopK': 20
}
```

### **main.py**
```python
from shutil import copyfile
import sys

sys.path.extend(["../"])
import params
from cmn import Common as cmn
cmn.logger=cmn.LogFile(f'../output/used_params_runid_{params.uml["RunId"]}.log')
from uml import UserSimilarities  as uml
from evl import ModelEvaluation as evl#, PytrecEvaluation as PE


def RunPipeline():
    cmn.logger.info(f'Main: UserSimilarities ...')
    copyfile('params.py', f'../output/used_params_runid_{params.uml["RunId"]}.py')
    uml.main(start=params.uml['start'],
             end=params.uml['end'],
             stopwords=['www', 'RT', 'com', 'http'],
             userModeling=params.uml['userModeling'],
             timeModeling=params.uml['timeModeling'],
             preProcessing=params.uml['preProcessing'],
             TagME=params.uml['TagME'],
             lastRowsNumber=params.uml['lastRowsNumber'], #10000, #all rows = 0
             num_topics=params.uml['num_topics'],
             filterExtremes=params.uml['filterExtremes'],
             library=params.uml['library'],
             path_2_save_tml=f'../output/{params.uml["RunId"]}/tml',
             path2_save_uml=f'../output/{params.uml["RunId"]}/uml',
             JO=params.uml['JO'],
             Bin=params.uml['Bin'],
             Threshold=params.uml['Threshold'],
             RunId=params.uml['RunId'])
    Pytrec_result = evl.main(RunId=params.evl['RunId'],
             path2_save_evl=f'../output/{params.evl["RunId"]}/evl',)
    return  Pytrec_result

PytrecResult = RunPipeline()


```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## References
<a id="1">[1]</a>  S.Ziaeinejad, H.Fani (2021). 
A Framework for Community Prediction in Microblogging
Social Networks.

