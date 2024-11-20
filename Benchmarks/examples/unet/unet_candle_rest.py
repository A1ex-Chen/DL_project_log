import candle
import unet
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint








if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:
        pass