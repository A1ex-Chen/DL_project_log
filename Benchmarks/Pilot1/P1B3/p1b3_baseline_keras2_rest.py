#! /usr/bin/env python

"""Multilayer Perceptron for drug response problem"""

from __future__ import division, print_function

import logging

# For non-interactive plotting
import matplotlib as mpl
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ProgbarLogger
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LocallyConnected1D,
    LocallyConnected2D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential

mpl.use("Agg")
import candle
import matplotlib.pyplot as plt
import p1b3 as benchmark
import tensorflow as tf

tf.compat.v1.disable_eager_execution()












class MyLossHistory(Callback):
    def __init__(
        self,
        progbar,
        val_gen,
        test_gen,
        val_steps,
        test_steps,
        metric,
        category_cutoffs=[0.0],
        ext="",
        pre="save",
    ):
        super(MyLossHistory, self).__init__()
        self.progbar = progbar
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.metric = metric
        self.category_cutoffs = category_cutoffs
        self.pre = pre
        self.ext = ext

    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf

    def on_epoch_end(self, batch, logs={}):
        val_loss, val_acc, y_true, y_pred, y_true_class, y_pred_class = evaluate_model(
            self.model, self.val_gen, self.val_steps, self.metric, self.category_cutoffs
        )
        test_loss, test_acc, _, _, _, _ = evaluate_model(
            self.model,
            self.test_gen,
            self.test_steps,
            self.metric,
            self.category_cutoffs,
        )
        self.progbar.append_extra_log_values(
            [("val_acc", val_acc), ("test_loss", test_loss), ("test_acc", test_acc)]
        )
        if float(logs.get("val_loss", 0)) < self.best_val_loss:
            plot_error(y_true, y_pred, batch, self.ext, self.pre)
        self.best_val_loss = min(float(logs.get("val_loss", 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get("val_acc", 0)), self.best_val_acc)


class MyProgbarLogger(ProgbarLogger):
    def __init__(self, samples):
        super(MyProgbarLogger, self).__init__(count_mode="steps")
        self.samples = samples
        self.params = {}

    def on_train_begin(self, logs=None):
        super(MyProgbarLogger, self).on_train_begin(logs)
        self.verbose = 1
        self.extra_log_values = []
        self.params["samples"] = self.samples
        self.params["metrics"] = []

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []
            self.extra_log_values = []

    def append_extra_log_values(self, tuples):
        for k, v in tuples:
            self.extra_log_values.append((k, v))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_log = "Epoch {}/{}".format(epoch + 1, self.epochs)
        for k in self.params["metrics"]:
            if k in logs:
                self.log_values.append((k, logs[k]))
                epoch_log += " - {}: {:.4f}".format(k, logs[k])
        for k, v in self.extra_log_values:
            self.log_values.append((k, v))
            epoch_log += " - {}: {:.4f}".format(k, float(v))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)
        benchmark.logger.debug(epoch_log)










class MyProgbarLogger(ProgbarLogger):






def add_conv_layer(model, layer_params, input_dim=None, locally_connected=False):
    if len(layer_params) == 3:  # 1D convolution
        filters = layer_params[0]
        filter_len = layer_params[1]
        stride = layer_params[2]
        if locally_connected:
            if input_dim:
                model.add(
                    LocallyConnected1D(
                        filters, filter_len, strides=stride, input_shape=(input_dim, 1)
                    )
                )
            else:
                model.add(LocallyConnected1D(filters, filter_len, strides=stride))
        else:
            if input_dim:
                model.add(
                    Conv1D(
                        filters, filter_len, strides=stride, input_shape=(input_dim, 1)
                    )
                )
            else:
                model.add(Conv1D(filters, filter_len, strides=stride))
    elif len(layer_params) == 5:  # 2D convolution
        filters = layer_params[0]
        filter_len = (layer_params[1], layer_params[2])
        stride = (layer_params[3], layer_params[4])
        if locally_connected:
            if input_dim:
                model.add(
                    LocallyConnected2D(
                        filters, filter_len, strides=stride, input_shape=(input_dim, 1)
                    )
                )
            else:
                model.add(LocallyConnected2D(filters, filter_len, strides=stride))
        else:
            if input_dim:
                model.add(
                    Conv2D(
                        filters, filter_len, strides=stride, input_shape=(input_dim, 1)
                    )
                )
            else:
                model.add(Conv2D(filters, filter_len, strides=stride))
    return model


def run(gParameters):
    """
    Runs the model using the specified set of parameters

    Args:
       gParameters: a python dictionary containing the parameters (e.g. epoch)
       to run the model with.
    """
    #
    if "dense" in gParameters:
        dval = gParameters["dense"]
        if type(dval) != list:
            res = list(dval)
            # try:
            #     is_str = isinstance(dval, basestring)
            # except NameError:
            #     is_str = isinstance(dval, str)
            # if is_str:
            #     res = str2lst(dval)
            gParameters["dense"] = res
        print(gParameters["dense"])

    if "conv" in gParameters:
        flat = gParameters["conv"]
        gParameters["conv"] = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        print("Conv input", gParameters["conv"])
    # print('Params:', gParameters)
    # Construct extension to save model
    ext = benchmark.extension_from_parameters(gParameters, ".keras")
    logfile = (
        gParameters["logfile"]
        if gParameters["logfile"]
        else gParameters["output_dir"] + ext + ".log"
    )

    fh = logging.FileHandler(logfile)
    fh.setFormatter(
        logging.Formatter(
            "[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(""))
    sh.setLevel(logging.DEBUG if gParameters["verbose"] else logging.INFO)

    benchmark.logger.setLevel(logging.DEBUG)
    benchmark.logger.addHandler(fh)
    benchmark.logger.addHandler(sh)
    benchmark.logger.info("Params: {}".format(gParameters))

    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()
    seed = gParameters["rng_seed"]

    # Build dataset loader object
    loader = benchmark.DataLoader(
        seed=seed,
        dtype=gParameters["data_type"],
        val_split=gParameters["val_split"],
        test_cell_split=gParameters["test_cell_split"],
        cell_features=gParameters["cell_features"],
        drug_features=gParameters["drug_features"],
        feature_subsample=gParameters["feature_subsample"],
        scaling=gParameters["scaling"],
        scramble=gParameters["scramble"],
        min_logconc=gParameters["min_logconc"],
        max_logconc=gParameters["max_logconc"],
        subsample=gParameters["subsample"],
        category_cutoffs=gParameters["category_cutoffs"],
    )

    # Initialize weights and learning rule
    initializer_weights = candle.build_initializer(
        gParameters["initialization"], kerasDefaults, seed
    )
    initializer_bias = candle.build_initializer("constant", kerasDefaults, 0.0)

    # Define model architecture
    gen_shape = None
    out_dim = 1

    model = Sequential()
    if "dense" in gParameters:  # Build dense layers
        for layer in gParameters["dense"]:
            if layer:
                model.add(
                    Dense(
                        layer,
                        input_dim=loader.input_dim,
                        kernel_initializer=initializer_weights,
                        bias_initializer=initializer_bias,
                    )
                )
                if gParameters["batch_normalization"]:
                    model.add(BatchNormalization())
                model.add(Activation(gParameters["activation"]))
                if gParameters["dropout"]:
                    model.add(Dropout(gParameters["dropout"]))
    else:  # Build convolutional layers
        gen_shape = "add_1d"
        layer_list = list(range(0, len(gParameters["conv"])))
        lc_flag = False
        if "locally_connected" in gParameters:
            lc_flag = True

        for _, i in enumerate(layer_list):
            if i == 0:
                add_conv_layer(
                    model,
                    gParameters["conv"][i],
                    input_dim=loader.input_dim,
                    locally_connected=lc_flag,
                )
            else:
                add_conv_layer(model, gParameters["conv"][i], locally_connected=lc_flag)
            if gParameters["batch_normalization"]:
                model.add(BatchNormalization())
            model.add(Activation(gParameters["activation"]))
            if gParameters["pool"]:
                model.add(MaxPooling1D(pool_size=gParameters["pool"]))
        model.add(Flatten())

    model.add(Dense(out_dim))

    # Define optimizer
    optimizer = candle.build_optimizer(
        gParameters["optimizer"], gParameters["learning_rate"], kerasDefaults
    )

    # Compile and display model
    model.compile(loss=gParameters["loss"], optimizer=optimizer)
    model.summary()
    benchmark.logger.debug("Model: {}".format(model.to_json()))

    train_gen = benchmark.DataGenerator(
        loader,
        batch_size=gParameters["batch_size"],
        shape=gen_shape,
        name="train_gen",
        cell_noise_sigma=gParameters["cell_noise_sigma"],
    ).flow()
    val_gen = benchmark.DataGenerator(
        loader,
        partition="val",
        batch_size=gParameters["batch_size"],
        shape=gen_shape,
        name="val_gen",
    ).flow()
    val_gen2 = benchmark.DataGenerator(
        loader,
        partition="val",
        batch_size=gParameters["batch_size"],
        shape=gen_shape,
        name="val_gen2",
    ).flow()
    test_gen = benchmark.DataGenerator(
        loader,
        partition="test",
        batch_size=gParameters["batch_size"],
        shape=gen_shape,
        name="test_gen",
    ).flow()

    train_steps = int(loader.n_train / gParameters["batch_size"])
    val_steps = int(loader.n_val / gParameters["batch_size"])
    test_steps = int(loader.n_test / gParameters["batch_size"])

    if "train_steps" in gParameters:
        train_steps = gParameters["train_steps"]
    if "val_steps" in gParameters:
        val_steps = gParameters["val_steps"]
    if "test_steps" in gParameters:
        test_steps = gParameters["test_steps"]

    checkpointer = ModelCheckpoint(
        filepath=gParameters["output_dir"] + ".model" + ext + ".h5", save_best_only=True
    )
    progbar = MyProgbarLogger(train_steps * gParameters["batch_size"])
    loss_history = MyLossHistory(
        progbar=progbar,
        val_gen=val_gen2,
        test_gen=test_gen,
        val_steps=val_steps,
        test_steps=test_steps,
        metric=gParameters["loss"],
        category_cutoffs=gParameters["category_cutoffs"],
        ext=ext,
        pre=gParameters["output_dir"],
    )

    # Seed random generator for training
    np.random.seed(seed)

    candleRemoteMonitor = candle.CandleRemoteMonitor(params=gParameters)

    # history = model.fit(train_gen, steps_per_epoch=train_steps, # this should be the deprecation fix
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=gParameters["epochs"],
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=0,
        callbacks=[checkpointer, loss_history, progbar, candleRemoteMonitor],
    )
    # callbacks=[checkpointer, loss_history, candleRemoteMonitor], # this just caused the job to hang on Biowulf

    benchmark.logger.removeHandler(fh)
    benchmark.logger.removeHandler(sh)

    return history


def main():

    gParameters = initialize_parameters()
    benchmark.check_params(gParameters)
    run(gParameters)


if __name__ == "__main__":
    main()
    try:
        K.clear_session()
    except AttributeError:  # theano does not have this function
        pass