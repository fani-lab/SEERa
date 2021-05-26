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
    # print('GraphClustering')
    # GC.GC_main()
    # print('NewsTopicExtraction')
    # NTE.NTE_main()
    # print('NewsRecommendation')
    # NR.NR_main()
    # print('ModelEvaluation')
    # ME.ME_main()
    # print('ModelEvaluation')
    # PE.PytrecEval_main()
    # print('Finished')

PytrecResult = RunPipeline()

