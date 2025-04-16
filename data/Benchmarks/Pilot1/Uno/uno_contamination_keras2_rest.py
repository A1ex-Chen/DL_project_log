#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import logging
import os

import candle
import numpy as np
import pandas as pd
import uno as benchmark
import uno_data
from joblib import dump
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoard,
)
from uno_baseline_keras2 import build_model, evaluate_prediction
from uno_data import (
    CombinedDataGenerator,
    CombinedDataLoader,
    DataFeeder,
    read_IDs_file,
)

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

additional_definitions = [
    {
        "name": "uq_exclude_drugs_file",
        "default": argparse.SUPPRESS,
        "action": "store",
        "help": "File with drug ids to exclude from training",
    },
    {
        "name": "uq_exclude_cells_file",
        "default": argparse.SUPPRESS,
        "action": "store",
        "help": "File with cell ids to exclude from training",
    },
    {
        "name": "uq_exclude_indices_file",
        "default": argparse.SUPPRESS,
        "action": "store",
        "help": "File with indices to exclude from training",
    },
    {
        "name": "exclude_drugs",
        "nargs": "+",
        "default": [],
        "help": "drug ids to exclude",
    },
    {
        "name": "exclude_cells",
        "nargs": "+",
        "default": [],
        "help": "cell ids to exclude",
    },
    {
        "name": "exclude_indices",
        "nargs": "+",
        "default": [],
        "help": "indices to exclude",
    },
    {
        "name": "reg_l2",
        "type": float,
        "default": 0.0,
        "help": "weight of regularization for l2 norm of nn weights",
    },
    {
        "name": "a_max",
        "type": float,
        "default": 0.99,
        "help": "maximum value admisible for a (global normal probability)",
    },
]

required = ["exclude_drugs", "exclude_cells", "exclude_indices"]











    df_pred_list = []

    cv_ext = ""
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info("Cross validation fold {}/{}:".format(fold + 1, cv))
            cv_ext = ".cv{}".format(fold + 1)
            prefix = prefix + "cv_" + str(fold)

        template_model = build_model(loader, args, silent=True)
        if args.initial_weights:
            logger.info("Loading initial weights from {}".format(args.initial_weights))
            template_model.load_weights(args.initial_weights)

        if len(args.gpus) > 1:
            from keras.utils import multi_gpu_model

            gpu_count = len(args.gpus)
            logger.info("Multi GPU with {} gpus".format(gpu_count))
            model = multi_gpu_model(template_model, cpu_merge=False, gpus=gpu_count)
        else:
            model = template_model

        optimizer = optimizers.deserialize({"class_name": args.optimizer, "config": {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)

        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params["timeout"])
        es_monitor = keras.callbacks.EarlyStopping(
            monitor="val_mae_contamination", patience=10, verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_mae_contamination", factor=0.5, patience=5, min_lr=0.00001
        )
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        checkpointer = candle.MultiGPUCheckpoint(
            prefix + cv_ext + ".model.h5", save_best_only=True
        )
        tensorboard = TensorBoard(
            log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext)
        )
        history_logger = candle.LoggingCallback(logger.debug)

        callbacks = [candle_monitor, timeout_monitor, history_logger]
        if args.es:
            callbacks.append(es_monitor)
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)
        if args.save_weights:
            logger.info("Will save weights to: " + args.save_weights)
            callbacks.append(candle.MultiGPUCheckpoint(args.save_weights))

        if args.use_exported_data is not None:
            train_gen = DataFeeder(
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
            )
            val_gen = DataFeeder(
                partition="val",
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
            )
            test_gen = DataFeeder(
                partition="test",
                filename=args.use_exported_data,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
                agg_dose=args.agg_dose,
            )
        else:
            train_gen = CombinedDataGenerator(
                loader,
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )
            val_gen = CombinedDataGenerator(
                loader,
                partition="val",
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )
            test_gen = CombinedDataGenerator(
                loader,
                partition="test",
                fold=fold,
                batch_size=args.batch_size,
                shuffle=args.shuffle,
                single=args.single,
            )

        df_val = val_gen.get_response(copy=True)
        y_val = df_val[target].values
        y_shuf = np.random.permutation(y_val)
        log_evaluation(
            evaluate_prediction(y_val, y_shuf),
            logger,
            description="Between random pairs in y_val:",
        )

        x_train_list, y_train = train_gen.get_slice(
            size=train_gen.size, single=args.single
        )
        x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)

        if y_train.ndim > 1:
            nout = y_train.shape[1]
        else:
            nout = 1

        logger.info("Training contamination model:")
        contamination_cbk = candle.Contamination_Callback(
            x_train_list, y_train, args.a_max
        )
        model.compile(
            loss=candle.contamination_loss(
                nout,
                contamination_cbk.T_k,
                contamination_cbk.a,
                contamination_cbk.sigmaSQ,
                contamination_cbk.gammaSQ,
            ),
            optimizer=optimizer,
            metrics=[
                candle.mae_contamination_metric(nout),
                candle.r2_contamination_metric(nout),
            ],
        )

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))

        callbacks.append(contamination_cbk)

        y_train_augmented = candle.add_index_to_output(y_train)
        y_val_aug_dum = candle.add_index_to_output(y_val)
        history = model.fit(
            x_train_list,
            y_train_augmented,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            validation_data=(x_val_list, y_val_aug_dum),
        )

        # prediction on holdout(test) when exists or use validation set
        if test_gen.size > 0:
            df_val = test_gen.get_response(copy=True)
            y_val = df_val[target].values
            y_val_pred = model.predict_generator(test_gen, test_gen.steps + 1)
            y_val_pred = y_val_pred[: test_gen.size]
        else:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)

        y_val_pred = y_val_pred.flatten()
        # df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred - y_val)
        df_val["Predicted" + target] = y_val_pred
        df_val[target + "Error"] = y_val_pred - y_val

        scores = evaluate_prediction(y_val, y_val_pred)
        log_evaluation(scores, logger)

        df_pred_list.append(df_val)

        if "loss" in history.history.keys():
            # Do not plot val loss since it is meaningless
            candle.plot_history(prefix, history, metric="loss", val=False)
        if "mae_contamination" in history.history.keys():
            candle.plot_history(prefix, history, metric="mae_contamination")
        if "r2_contamination" in history.history.keys():
            candle.plot_history(prefix, history, metric="r2_contamination")

        # Plot a evolution
        fname = prefix + ".evol.a.png"
        xlabel = "Epochs"
        ylabel = "Contamination a"
        title = "a Evolution"
        candle.plot_array(contamination_cbk.avalues, xlabel, ylabel, title, fname)
        # Plot sigmaSQ evolution
        fname = prefix + ".evol.sigmasq.png"
        xlabel = "Epochs"
        ylabel = "Contamination SigmaSQ"
        title = "SigmaSQ Evolution"
        candle.plot_array(contamination_cbk.sigmaSQvalues, xlabel, ylabel, title, fname)
        # Plot gammaSQ evolution
        fname = prefix + ".evol.gammasq.png"
        xlabel = "Epochs"
        ylabel = "Contamination GammaSQ"
        title = "GammaSQ Evolution"
        candle.plot_array(contamination_cbk.gammaSQvalues, xlabel, ylabel, title, fname)
        # Plot latent variables and outliers
        sigma = np.sqrt(K.get_value(contamination_cbk.sigmaSQ))
        gamma = np.sqrt(K.get_value(contamination_cbk.gammaSQ))
        T = K.get_value(contamination_cbk.T_k)
        dictCont = {"sigma": sigma, "gamma": gamma, "T": T}
        cpar_fname = prefix + ".contPar.joblib"
        dump(dictCont, cpar_fname)

        y_tr_pred = model.predict(x_train_list, batch_size=args.batch_size)
        candle.plot_contamination(
            y_train, y_tr_pred.squeeze(), sigma, T, pred_name=target, figprefix=prefix
        )

    pred_fname = prefix + ".predicted.tsv"
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        if args.single:
            df_pred.sort_values(["Sample", "Drug1", target], inplace=True)
        else:
            df_pred.sort_values(
                ["Source", "Sample", "Drug1", "Drug2", target], inplace=True
            )
    else:
        if args.single:
            df_pred.sort_values(["Sample", "Drug1", "Dose1", "Growth"], inplace=True)
        else:
            df_pred.sort_values(
                ["Sample", "Drug1", "Drug2", "Dose1", "Dose2", "Growth"], inplace=True
            )
    df_pred.to_csv(pred_fname, sep="\t", index=False, float_format="%.4g")

    if args.cv > 1:
        scores = evaluate_prediction(df_pred[target], df_pred["Predicted" + target])
        log_evaluation(scores, description="Combining cross validation folds:")

    for test_source in loader.test_sep_sources:
        test_gen = CombinedDataGenerator(
            loader, partition="test", batch_size=args.batch_size, source=test_source
        )
        df_test = test_gen.get_response(copy=True)
        y_test = df_test[target].values
        n_test = len(y_test)
        if n_test == 0:
            continue

        x_test_list, y_test = test_gen.get_slice(size=test_gen.size, single=args.single)
        y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
        y_test_pred = y_test_pred.flatten()
        scores = evaluate_prediction(y_test, y_test_pred)
        log_evaluation(
            scores,
            logger,
            description="Testing on data from {} ({})".format(test_source, n_test),
        )

    if K.backend() == "tensorflow":
        K.clear_session()

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == "__main__":
    main()
    if K.backend() == "tensorflow":
        K.clear_session()