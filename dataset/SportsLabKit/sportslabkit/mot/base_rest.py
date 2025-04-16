from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np
import optuna
import pandas as pd

from sportslabkit import Tracklet
from sportslabkit.detection_model.dummy import DummyDetectionModel
from sportslabkit.logger import logger, tqdm
from sportslabkit.metrics import hota_score



    return wrapper


class Callback:
    """Base class for creating new callbacks.

    This class defines the basic structure of a callback and allows for dynamic method creation
    for handling different events in the Trainer's lifecycle.

    Methods:
        __getattr__(name: str) -> callable:
            Returns a dynamically created method based on the given name.
    """

    pass

class MultiObjectTracker(ABC):




    @abstractmethod



    @with_callbacks












    @property

    @property

    @property





        tracklets = list(filter(filter_short_tracklets, tracklets))
        return tracklets

    def increment_staleness(self, tracklets):
        for i, _ in enumerate(tracklets):
            tracklets[i].staleness += 1
        return tracklets

    def reset_staleness(self, tracklets):
        for i, _ in enumerate(tracklets):
            tracklets[i].staleness = 0
        return tracklets

    def pre_track(self):
        # Hook that subclasses can override
        pass

    def post_track(self):
        pass

    def reset(self):
        # Initialize the single object tracker
        logger.debug("Initializing tracker...")
        self.alive_tracklets = []
        self.dead_tracklets = []
        self.frame_count = 0
        logger.debug("Tracker initialized.")

    def _check_required_observations(self, target: dict[str, Any]):
        missing_types = [
            required_type for required_type in self.required_observation_types if required_type not in target
        ]

        if missing_types:
            required_types_str = ", ".join(self.required_observation_types)
            missing_types_str = ", ".join(missing_types)
            current_types_str = ", ".join(target.keys())

            raise ValueError(
                f"Input 'target' is missing the following required types: {missing_types_str}.\n"
                f"Required types: {required_types_str}\n"
                f"Current types in 'target': {current_types_str}"
            )

    def check_updated_state(self, state: dict[str, Any]):
        if not isinstance(state, dict):
            raise ValueError("The `update` method must return a dictionary.")

        missing_types = [
            required_type for required_type in self.required_observation_types if required_type not in state
        ]

        if missing_types:
            missing_types_str = ", ".join(missing_types)
            raise ValueError(
                f"The returned state from `update` is missing the following required types: {missing_types_str}."
            )

    def create_tracklet(self, state: dict[str, Any]):
        tracklet = Tracklet(max_staleness=self.max_staleness)
        for required_type in self.required_observation_types:
            tracklet.register_observation_type(required_type)
        for required_type in self.required_state_types:
            tracklet.register_state_type(required_type)
        self._check_required_observations(state)
        self.update_tracklet(tracklet, state)
        return tracklet

    def to_bbdf(self):
        """Create a bounding box dataframe."""
        all_tracklets = self.alive_tracklets + self.dead_tracklets
        return pd.concat([t.to_bbdf() for t in all_tracklets], axis=1).sort_index()

    def separate_stale_tracklets(self, unassigned_tracklets):
        stale_tracklets, non_stale_tracklets = [], []
        for tracklet in unassigned_tracklets:
            if tracklet.is_stale():
                stale_tracklets.append(tracklet)
            else:
                non_stale_tracklets.append(tracklet)
        return non_stale_tracklets, stale_tracklets

    @property
    def required_observation_types(self):
        raise NotImplementedError

    @property
    def required_state_types(self):
        raise NotImplementedError

    @property
    def hparam_searh_space(self):
        return {}

    def create_hparam_dict(self):
        hparam_search_space = {}
        # Create a dictionary for all hyperparameters
        hparams = {"self": self.hparam_search_space} if hasattr(self, "hparam_search_space") else {}
        for attribute in vars(self):
            value = getattr(self, attribute)
            if hasattr(value, "hparam_search_space") and attribute not in hparam_search_space:
                hparams[attribute] = {}
                search_space = value.hparam_search_space
                for param_name, param_space in search_space.items():
                    hparams[attribute][param_name] = {
                        "type": param_space["type"],
                        "values": param_space.get("values"),
                        "low": param_space.get("low"),
                        "high": param_space.get("high"),
                    }
        return hparams

    def get_new_hyperparameters(self, hparams, trial):
        params = {}
        for attribute, param_space in hparams.items():
            params[attribute] = {}
            for param_name, param_values in param_space.items():
                if param_values["type"] == "categorical":
                    params[attribute][param_name] = trial.suggest_categorical(param_name, param_values["values"])
                elif param_values["type"] == "float":
                    params[attribute][param_name] = trial.suggest_float(
                        param_name, param_values["low"], param_values["high"]
                    )
                elif param_values["type"] == "logfloat":
                    params[attribute][param_name] = trial.suggest_float(
                        param_name,
                        param_values["low"],
                        param_values["high"],
                        log=True,
                    )
                elif param_values["type"] == "int":
                    params[attribute][param_name] = trial.suggest_int(
                        param_name, param_values["low"], param_values["high"]
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_values['type']}")
        return params

    def apply_hyperparameters(self, params):
        # Apply the hyperparameters to the attributes of `self`
        for attribute, param_values in params.items():
            for param_name, param_value in param_values.items():
                if attribute not in self.__dict__ and attribute != "self":
                    raise AttributeError(f"{attribute=} not found in object")  # Raising specific error

                if attribute == "self":
                    logger.debug(f"Setting {param_name} to {param_value} for {self}")
                    setattr(self, param_name, param_value)
                else:
                    attr_obj = getattr(self, attribute)
                    if param_name in attr_obj.__dict__:
                        setattr(attr_obj, param_name, param_value)
                        logger.debug(f"Setting {param_name} to {param_value} for {attribute}")
                    else:
                        __dict__ = attr_obj.__dict__
                        raise TypeError(
                            f"Cannot set {param_name=} on {attribute=}, as it is immutable or not in {list(__dict__.keys())}"
                        )

    def tune_hparams(
        self,
        frames_list,
        bbdf_gt_list,
        n_trials=100,
        hparam_search_space=None,
        verbose=False,
        return_study=False,
        use_bbdf=False,
        reuse_detections=False,
        sampler=None,
        pruner=None,
    ):

        hparams = hparam_search_space or self.create_hparam_dict()

        logger.info("Hyperparameter search space:")
        for attribute, param_space in hparams.items():
            logger.info(f"{attribute}:")
            for param_name, param_values in param_space.items():
                logger.info(f"\t{param_name}: {param_values}")
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        if use_bbdf:
            raise NotImplementedError
        if reuse_detections:
            self.detection_models = []

            for frames in frames_list:
                list_of_detections = []
                for frame in tqdm(frames, desc="Detecting frames for reuse"):
                    list_of_detections.append(self.detection_model(frame)[0])

                dummy_detection_model = DummyDetectionModel(list_of_detections)
                og_detection_model = self.detection_model
                self.detection_models.append(dummy_detection_model)

        if sampler is None:
            sampler = optuna.samplers.TPESampler(multivariate=True)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner()

        self.trial_params = []  # Used to store the parameters for each trial
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=n_trials)

        if reuse_detections:
            # reset detection model
            self.detection_model = og_detection_model

        best_value = study.best_value
        self.best_params = self.trial_params[study.best_trial.number]
        self.apply_hyperparameters(self.best_params)

        if return_study:
            return self.best_params, best_value, study
        return self.best_params, best_value