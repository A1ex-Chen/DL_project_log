# Setup

import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import smiles_transformer as st
import tensorflow.config.experimental

gpus = tensorflow.config.experimental.list_physical_devices("GPU")
try:
    for gpu in gpus:
        print("setting memory growth")
        tensorflow.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)




# Train and Evaluate






if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()