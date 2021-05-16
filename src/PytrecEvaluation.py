"""A very simple example."""

import pytrec_eval
import json
import pickle


def load_obj(name):
	with open(name+'.pkl', 'rb') as f:
		return pickle.load(f)

def PytrecEval_main():
    qrel = {
        'n1': {
            'c3': 1,
            'c40': 1,
        },
        'n2': {
            'c2': 1,
            'c3': 1,
        },
    }

    run = {
        'n1': {

        },
        'n2': {
            'c1': 1,
            'c2': 1,
            'c3': 1,
        }
    }
    qrel = load_obj('../RecommendedNews_UserBased')
    run  = load_obj('../MentionedNews_UserBased')
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'success_1', 'success_5', 'success_10'})
    output = evaluator.evaluate(run)
    print(json.dumps(output, indent=1))
    with open('../Pytrec_eval.txt', 'w') as outfile:
        json.dump(output, outfile)
