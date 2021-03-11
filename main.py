import os
import numpy as np
import DataReader
import DataPreparation
import TopicModeling
import ToyGraphGeneration
import UserSimilarities
import UsersGraph
import graphEmbedding
import GraphReconstruction
import GraphClustering


if __name__ == '__main__':
    scenario = 'scenario' + str(11)
    os.chdir(scenario)
