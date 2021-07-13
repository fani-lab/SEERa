
# Community Prediction in Microblogging Social Networks

This is an open-source python-based framework to predict the future user communities in a text streaming social network (e.g., Twitter) based on the users’ topics of interest.

## Installation

It is strongly recommended to use Linux OS for installing the packages and executing the framework. To install packages and dependencies, simply use this command in your shell:

```bash
 pip install -r requirements.txt
```

## Usage

### Framework Structure:
Our framework has six major layers: Data Access Layer (DAL),
Topic Modeling Layer (TML), User Modeling Layer (UML), Graph
Embedding Layer (GEL), and Community Prediction Layer (CPL).
The application layer, which is described in Section 3, is the last
layer to show how our method improves the performance of an application. [[1]](#1).\
...\
├── **output**\
├── **src**\
│&nbsp;&nbsp;&nbsp;\
│&nbsp;&nbsp;&nbsp;├── **cmn** (common functions)\
│&nbsp;&nbsp;&nbsp;├──── *Common.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **dal**  (data access layer)\
│&nbsp;&nbsp;&nbsp;├──── *DataPreparation.py*\
│&nbsp;&nbsp;&nbsp;├──── *DataReader.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **tml**  (topic modeling layer)\
│&nbsp;&nbsp;&nbsp;├──── *TopicModeling.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **uml** (user modeling layer)\
│&nbsp;&nbsp;&nbsp;├──── *UsersGraph.py*\
│&nbsp;&nbsp;&nbsp;├──── *UserSimilarities.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **gel** (graph embedding layer)\
│&nbsp;&nbsp;&nbsp;├──── *GraphEmbedding.py*\
│&nbsp;&nbsp;&nbsp;├──── *GraphReconstruction.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **cpl** (community prediction layer)\
│&nbsp;&nbsp;&nbsp;├──── *GraphClustering.py*\
│&nbsp;&nbsp;&nbsp;├─\
│&nbsp;&nbsp;&nbsp;├── **application layer**\
│&nbsp;&nbsp;&nbsp;├──── *NewsTopicExtraction.py*\
│&nbsp;&nbsp;&nbsp;├──── *NewsRecommendation.py*\
│&nbsp;&nbsp;&nbsp;├──── *ModelEvaluation.py*




```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## References
[1] Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

## License
[MIT](https://choosealicense.com/licenses/mit/)