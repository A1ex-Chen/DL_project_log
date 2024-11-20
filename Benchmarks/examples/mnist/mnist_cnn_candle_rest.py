# from tensorflow.keras.callbacks import CSVLogger
import candle
import mnist
from tensorflow.keras import backend as K








if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:
        pass