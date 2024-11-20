import json
import os
from datetime import datetime

import numpy as np
from tensorflow.keras.callbacks import Callback




class CandleRemoteMonitor(Callback):
    """Capture Run level output and store/send for monitoring"""








class TerminateOnTimeOut(Callback):
    """This class implements timeout on model training. When the script reaches timeout,
    this class sets model.stop_training = True
    """


