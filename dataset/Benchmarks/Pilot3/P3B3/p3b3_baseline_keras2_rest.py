from __future__ import print_function

import candle
import keras_mt_shared_cnn
import numpy as np
import p3b3 as bmk
from tensorflow.keras import backend as K

# from tensorflow.keras.layers import Input, Dense, Dropout, Activation
# from tensorflow.keras.optimizers import SGD, Adam, RMSprop
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# from sklearn.metrics import f1_score


# import keras












if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass