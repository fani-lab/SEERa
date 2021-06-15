"""A very simple example."""

# import pytrec_eval
import json
import pickle
import params
import numpy as np
from cmn import Common as cmn



#
# def main(qrel, run):
#     # qrel = load_obj('../RecommendedNews_UserBased')
#     # run  = load_obj('../MentionedNews_UserBased')
#     evaluator = pytrec_eval.RelevanceEvaluator(
#         qrel, {'success_1', 'success_5', 'success_10', 'success_100'})
#     output = evaluator.evaluate(run)
#     with open(f'../output/{params.evl["RunId"]}/evl/Pytrec_eval.txt', 'w') as outfile:
#         json.dump(output, outfile)


def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)


def main(recom, mention):
    # qrel = load_obj(recommendation)
    # run = load_obj(mentions)
    cmn.logger.info(f'\n\nEvaluation:\n')
    Mentioners = 0
    hitCounter = 0
    All_Users_number = 0
    for user in range(len(recom.keys())):
        All_Users_number += 1
        if len(mention[f'u{user+1}'].keys()) > 0:
            Mentioners += 1
            # Flag = False
            for i in recom[f'u{user+1}'].keys():
                if i in mention[f'u{user+1}'].keys():
                    hitCounter += 1
                    # Flag = True
                    break
            # if not Flag:
            #     print(recom[f'u{user+1}'].keys())
            #     print(mention[f'u{user+1}'].keys())
            #     print('------------------------------------')

    print('All Users:', All_Users_number)
    print('hits:', hitCounter)
    cmn.logger.info(f'Evaluation: hits: {hitCounter}')
    print('percentage:', hitCounter/Mentioners)
    cmn.logger.info(f'Evaluation: percentage: {hitCounter/Mentioners}')
    print('topK recommendations:', len(recom['u1']))
    cmn.logger.info(f'Evaluation: topK recommendations: {len(recom["u1"])}')
    print('users:', len(recom.keys()))
    cmn.logger.info(f'Evaluation: users: {len(recom.keys())}')
    return hitCounter

def main2(recom, mention):

    cmn.logger.info(f'\n\nEvaluation:\n')
    hitCounter = 0
    Mentioners = 0
    for user in range(len(recom)):
        if len(mention[user]) > 0:
            Mentioners += 1
            if len(np.intersect1d(recom[user], mention[user])) > 0:
                hitCounter += 1
    print('main2All Users:', len(recom))
    print('main2hits:', hitCounter)
    cmn.logger.info(f'Evaluation: hits: {hitCounter}')
    print('main2percentage:', hitCounter/Mentioners)
    cmn.logger.info(f'Evaluation: percentage: {hitCounter/Mentioners}')
    print('main2topK recommendations:', len(recom[0]))
    cmn.logger.info(f'Evaluation: topK recommendations: {len(recom[0])}')
    print('main2users:', len(mention))
    cmn.logger.info(f'Evaluation: users: {len(mention)}')
    return hitCounter