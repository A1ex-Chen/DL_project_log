from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np
import optuna

from sportslabkit import Tracklet
from sportslabkit.dataframe.bboxdataframe import BBoxDataFrame
from sportslabkit.logger import logger
from sportslabkit.metrics.object_detection import iou_scores


class SingleObjectTracker(ABC):




    @abstractmethod








    @property

    @property



        # check that the ground truth positions are in the correct format
        if isinstance(ground_truth_positions, BBoxDataFrame):
            ground_truth_positions = np.expand_dims(ground_truth_positions.values, axis=1)[:, :, :4]

        hparams = self.create_hparam_dict()

        print("Hyperparameter search space: ")
        for attribute, param_space in hparams.items():
            print(f"{attribute}:")
            for param_name, param_values in param_space.items():
                print(f"\t{param_name}: {param_values}")
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_iou = study.best_value
        if return_study:
            return best_params, best_iou, study
        return best_params, best_iou