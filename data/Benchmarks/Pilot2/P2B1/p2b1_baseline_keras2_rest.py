import logging
import os
import sys

import numpy as np

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

TIMEOUT = 3600  # in sec; set this to -1 for no timeout
file_path = os.path.dirname(os.path.realpath(__file__))

import candle
import p2b1
import p2b1_AE_models as AE_models
from tensorflow.keras import backend as K

HOME = os.environ["HOME"]

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"











    # lr_scheduler = LearningRateScheduler(step_decay)
    history = callbacks.History()
    # callbacks=[history,lr_scheduler]

    history_logger = candle.LoggingCallback(logger.debug)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=GP)
    timeoutMonitor = candle.TerminateOnTimeOut(TIMEOUT)
    callbacks = [history, history_logger, candleRemoteMonitor, timeoutMonitor]
    # loss = 0.

    # ### Save the Model to disk
    if GP["save_path"] is not None:
        save_path = GP["save_path"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = "."

    model_json = molecular_model.to_json()
    with open(save_path + "/model.json", "w") as json_file:
        json_file.write(model_json)

    encoder_json = molecular_encoder.to_json()
    with open(save_path + "/encoder.json", "w") as json_file:
        json_file.write(encoder_json)

    print("Saved model to disk")

    # ### Train the Model
    if GP["train_bool"]:
        ct = hf.Candle_Molecular_Train(
            molecular_model,
            molecular_encoder,
            data_files,
            mb_epochs,
            callbacks,
            batch_size=batch_size,
            nbr_type=GP["nbr_type"],
            save_path=GP["save_path"],
            len_molecular_hidden_layers=len_molecular_hidden_layers,
            molecular_nbrs=molecular_nbrs,
            conv_bool=conv_bool,
            full_conv_bool=full_conv_bool,
            type_bool=GP["type_bool"],
            sampling_density=GP["sampling_density"],
        )
        frame_loss, frame_mse = ct.train_ac()
    else:
        frame_mse = []
        frame_loss = []

    return frame_loss, frame_mse


def main():

    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass