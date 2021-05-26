"""A very simple example."""

import pytrec_eval
import json
import pickle
import params


def load_obj(name):
	with open(name+'.pkl', 'rb') as f:
		return pickle.load(f)

def main(qrel, run):
    # qrel = load_obj('../RecommendedNews_UserBased')
    # run  = load_obj('../MentionedNews_UserBased')
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'success_1', 'success_5', 'success_10'})
    output = evaluator.evaluate(run)
    return json.dumps(output, indent=1)
    with open(f'../output/{params.evl["RunId"]}/evl/Pytrec_eval.txt', 'w') as outfile:
        json.dump(output, outfile)
