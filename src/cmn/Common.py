import logging
import sys

sys.path.extend(["../"])
import params

def LogFile(file='logfile.log'):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if file:
        file_handler = logging.FileHandler(filename=file, mode='w')
        logger.addHandler(file_handler)

    return logger

logger=LogFile()