import numpy as np
import optuna

from sportslabkit import Tracklet
from sportslabkit.metrics import convert_to_x1y1x2y2, iou_score



    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if hparam_search_space is None:
        hparam_search_space = motion_model_class.hparam_search_space

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_iou = 1 - study.best_value
    if return_study:
        return best_params, best_iou, study
    return best_params, best_iou