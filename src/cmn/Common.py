import logging
import sys
import numpy as np

sys.path.extend(["../"])
import params

def save2excel(array, savename):
    np.savetxt(f"../output/{params.uml['RunId']}/{savename}.csv", array, delimiter=",", fmt='%s', encoding='utf-8')

def LogFile(file='logfile.log'):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    if file:
        file_handler = logging.FileHandler(filename=file, mode='w')
        logger.addHandler(file_handler)

    return logger

logger=LogFile()