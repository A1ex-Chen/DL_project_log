"""Tensorflow trainer class."""

import datetime
import math
import os
from typing import Callable, Dict, Optional, Tuple


# Integrations must be imported before ML frameworks:
from .integrations import (  # isort: split
    is_comet_available,
    is_wandb_available,
)

import numpy as np
import tensorflow as tf
from packaging.version import parse
from tensorflow.python.distribute.values import PerReplica

from .modeling_tf_utils import TFPreTrainedModel
from .optimization_tf import GradientAccumulator, create_optimizer
from .trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, EvaluationStrategy, PredictionOutput, set_seed
from .training_args_tf import TFTrainingArguments
from .utils import logging


if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

logger = logging.get_logger(__name__)


class TFTrainer:
    """
    TFTrainer is a simple but feature-complete training and eval loop for TensorFlow, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.TFPreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TFTrainingArguments`):
            The arguments to tweak training.
        train_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for training. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        eval_dataset (:class:`~tf.data.Dataset`, `optional`):
            The dataset to use for evaluation. The dataset should yield tuples of ``(features, labels)`` where
            ``features`` is a dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss
            is calculated by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as
            when using a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
            ``model(features, **labels)``.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        tb_writer (:obj:`tf.summary.SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule]`, `optional`):
            A tuple containing the optimizer and the scheduler to use. The optimizer default to an instance of
            :class:`tf.keras.optimizers.Adam` if :obj:`args.weight_decay_rate` is 0 else an instance of
            :class:`~transformers.AdamWeightDecay`. The scheduler will default to an instance of
            :class:`tf.keras.optimizers.schedules.PolynomialDecay` if :obj:`args.num_warmup_steps` is 0 else an
            instance of :class:`~transformers.WarmUp`.
    """












    @tf.function




    @tf.function

    @staticmethod

    @staticmethod


