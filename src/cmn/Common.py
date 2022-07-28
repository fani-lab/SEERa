import logging
import sys
import numpy as np
import random

random.seed(0)
np.random.seed(0)

sys.path.extend(["../"])

def LogFile(file='logfile.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if file:
        file_handler = logging.FileHandler(filename=file, mode='w')
        logger.addHandler(file_handler)
    return logger

logger=None#LogFile()