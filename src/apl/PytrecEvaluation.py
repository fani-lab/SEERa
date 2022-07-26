import pytrec_eval
import json
import pickle
import params
import numpy as np
from cmn import Common as cmn




def main(qrel, run):
    #qrel = load_obj('../RecommendedNews_UserBased')
    #run  = load_obj('../MentionedNews_UserBased')
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'success_1', 'success_5', 'success_10', 'success_100'})
    output = evaluator.evaluate(run)
    with open(f'{params.apl["path2save"]}/evl/Pytrec_eval.txt', 'w') as outfile:
        json.dump(output, outfile)


def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)


def main2(recom, mention):
    cmn.logger.info(f'\n\nEvaluation:\n')
    mentioners = 0
    hit_counter = 0
    all_users_number = 0
    for user in recom:
        all_users_number += 1
        if len(mention[user].keys()) > 0:
            mentioners += 1
            for i in recom[user].keys():
                if i in mention[user].keys():
                    hit_counter += 1
                    break

    print('All Users:', all_users_number)
    print('hits:', hit_counter)
    cmn.logger.info(f'Evaluation: hits: {hit_counter}')
    print('percentage:', hit_counter/mentioners)
    cmn.logger.info(f'Evaluation: percentage: {hit_counter/mentioners}')
    print('topK recommendations:', len(recom['u1']))
    cmn.logger.info(f'Evaluation: topK recommendations: {len(recom["u1"])}')
    print('users:', len(recom.keys()))
    cmn.logger.info(f'Evaluation: users: {len(recom.keys())}')
    return hit_counter

def main3(recom, mention):
    cmn.logger.info(f'\n\nEvaluation:\n')
    hit_counter = 0
    mentioners = 0
    for user in mention:
        if len(mention[user]) > 0:
            mentioners += 1
            if len(np.intersect1d(recom[user].keys(), mention[user].keys())) > 0:
                hit_counter += 1
    print('main2All Users:', len(recom))
    print('main2hits:', hit_counter)
    cmn.logger.info(f'Evaluation: hits: {hit_counter}')
    print('main2percentage:', hit_counter/mentioners)
    cmn.logger.info(f'Evaluation: percentage: {hit_counter/mentioners}')
    print('main2users:', len(mention))
    cmn.logger.info(f'Evaluation: users: {len(mention)}')
    return hit_counter
