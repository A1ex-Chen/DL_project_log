import logging

logger = logging.getLogger("DARTS")
fh = logging.FileHandler("darts_accuracy.log")
logger.addHandler(fh)



