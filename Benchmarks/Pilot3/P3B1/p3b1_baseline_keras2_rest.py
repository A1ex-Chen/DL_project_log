from __future__ import print_function

import candle
import numpy as np
import p3b1 as bmk
from sklearn.metrics import f1_score
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
















if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass