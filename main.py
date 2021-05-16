import src.UserSimilarities    as US
import src.GraphClustering     as GC
import src.NewsTopicExtraction as NTE
import src.NewsRecommendation  as NR
import src.ModelEvaluation     as ME
import src.PytrecEvaluation    as PE

def RunPipeline():
    print('UserSimilarities')
    US.US_main()
    print('GraphClustering')
    GC.GC_main()
    print('NewsTopicExtraction')
    NTE.NTE_main()
    print('NewsRecommendation')
    NR.NR_main()
    print('ModelEvaluation')
    ME.ME_main()
    print('ModelEvaluation')
    PE.PytrecEval_main()
    print('Finished')

RunPipeline()