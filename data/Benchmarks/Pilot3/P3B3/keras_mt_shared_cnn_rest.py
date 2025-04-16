"""
Code to export keras architecture/placeholder weights for MT CNN
Written by Mohammed Alawad
Date: 10_20_2017
"""
# from tensorflow.keras.layers.convolutional import Conv1D
from tensorflow.keras.layers import (
    Concatenate,
    Convolution1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)

# np.random.seed(1337)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2




if __name__ == "__main__":
    main()