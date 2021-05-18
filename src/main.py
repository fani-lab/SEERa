from shutil import copyfile
import sys

sys.path.extend(["../"])
from cmn import Common as cmn
import params
from uml import UserSimilarities  as uml
# import GraphClustering     as GC
# import NewsTopicExtraction as NTE
# import NewsRecommendation  as NR
# import ModelEvaluation     as ME
# import PytrecEvaluation    as PE

def RunPipeline():
    cmn.logger.info(f'Main: UserSimilarities ...')
    copyfile('params.py', f'../output/used_params_runid_{params.uml["runid"]}.py')
    uml.main(start=params.uml['start'],
             end=params.uml['end'],
             stopwords=['www', 'RT', 'com', 'http'],
             userModeling=params.uml['userModeling'],
             timeModeling=params.uml['timeModeling'],
             preProcessing=params.uml['preProcessing'],
             TagME=params.uml['TagME'], 
             lastRowsNumber=0, #all rows
             num_topics=params.uml['num_topics'],
             filterExtremes=params.uml['filterExtremes'],
             library=params.uml['library'],
             path_2_save_tml=f'../output/{params.uml["runid"]}/tml',
             path2_save_uml=f'../output/{params.uml["runid"]}/uml',
             JO=params.uml['JO'],
             Bin=params.uml['Bin'],
             Threshold=params.uml['Threshold'])

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

RunPipeline()