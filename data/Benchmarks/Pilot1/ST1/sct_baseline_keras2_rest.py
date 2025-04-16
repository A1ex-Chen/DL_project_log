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




# Train and Evaluate






if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()