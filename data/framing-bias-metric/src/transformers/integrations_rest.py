# Integrations with other Python libraries
import math
import os

from .trainer_utils import EvaluationStrategy
from .utils import logging


logger = logging.get_logger(__name__)


# Import 3rd-party integrations before ML frameworks:

try:
    # Comet needs to be imported before any ML frameworks
    import comet_ml  # noqa: F401

    if hasattr(comet_ml, "config") and comet_ml.config.get_config("comet.api_key"):
        _has_comet = True
    else:
        if os.getenv("COMET_MODE", "").upper() != "DISABLED":
            logger.warning("comet_ml is installed but `COMET_API_KEY` is not set.")
        _has_comet = False
except (ImportError, ValueError):
    _has_comet = False

try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        if os.getenv("WANDB_DISABLED"):
            logger.warning("W&B installed but not logged in. Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False

try:
    import optuna  # noqa: F401

    _has_optuna = True
except (ImportError):
    _has_optuna = False

try:
    import ray  # noqa: F401

    _has_ray = True
except (ImportError):
    _has_ray = False

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # noqa: F401

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False

try:
    from azureml.core.run import Run  # noqa: F401

    _has_azureml = True
except ImportError:
    _has_azureml = False

try:
    import mlflow  # noqa: F401

    _has_mlflow = True
except ImportError:
    _has_mlflow = False

# No transformer imports above this point

from .file_utils import is_torch_tpu_available  # noqa: E402
from .trainer_callback import TrainerCallback  # noqa: E402
from .trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun  # noqa: E402


# Integration functions:























    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)


def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:

    # The model and TensorBoard writer do not pickle so we have to remove them (if they exists)
    # while doing the ray hp search.

    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None
    # Setup default `resources_per_trial` and `reporter`.
    if "resources_per_trial" not in kwargs and trainer.args.n_gpu > 0:
        # `args.n_gpu` is considered the total number of GPUs that will be split
        # among the `n_jobs`
        n_jobs = int(kwargs.pop("n_jobs", 1))
        num_gpus_per_trial = trainer.args.n_gpu
        if num_gpus_per_trial / n_jobs >= 1:
            num_gpus_per_trial = int(math.ceil(num_gpus_per_trial / n_jobs))
        kwargs["resources_per_trial"] = {"gpu": num_gpus_per_trial}

    if "progress_reporter" not in kwargs:
        from ray.tune import CLIReporter

        kwargs["progress_reporter"] = CLIReporter(metric_columns=["objective"])
    if "keep_checkpoints_num" in kwargs and kwargs["keep_checkpoints_num"] > 0:
        # `keep_checkpoints_num=0` would disabled checkpointing
        trainer.use_tune_checkpoints = True
        if kwargs["keep_checkpoints_num"] > 1:
            logger.warning(
                "Currently keeping {} checkpoints for each trial. Checkpoints are usually huge, "
                "consider setting `keep_checkpoints_num=1`."
            )
    if "scheduler" in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

        # Check if checkpointing is enabled for PopulationBasedTraining
        if isinstance(kwargs["scheduler"], PopulationBasedTraining):
            if not trainer.use_tune_checkpoints:
                logger.warning(
                    "You are using PopulationBasedTraining but you haven't enabled checkpointing. "
                    "This means your trials will train from scratch everytime they are exploiting "
                    "new configurations. Consider enabling checkpointing by passing "
                    "`keep_checkpoints_num=1` as an additional argument to `Trainer.hyperparameter_search`."
                )

        # Check for `do_eval` and `eval_during_training` for schedulers that require intermediate reporting.
        if isinstance(
            kwargs["scheduler"], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)
        ) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == EvaluationStrategy.NO):
            raise RuntimeError(
                "You are using {cls} as a scheduler but you haven't enabled evaluation during training. "
                "This means your trials will not report intermediate results to Ray Tune, and "
                "can thus not be stopped early or used to exploit other trials parameters. "
                "If this is what you want, do not use {cls}. If you would like to use {cls}, "
                "make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the "
                "Trainer `args`.".format(cls=type(kwargs["scheduler"]).__name__)
            )

    analysis = ray.tune.run(_objective, config=trainer.hp_space(None), num_samples=n_trials, **kwargs)
    best_trial = analysis.get_best_trial(metric="objective", mode=direction[:3])
    best_run = BestRun(best_trial.trial_id, best_trial.last_result["objective"], best_trial.config)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class TensorBoardCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `TensorBoard
    <https://www.tensorflow.org/tensorboard>`__.

    Args:
        tb_writer (:obj:`SummaryWriter`, `optional`):
            The writer to use. Will instantiate one if not set.
    """







class WandbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """






class CometCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Comet ML <https://www.comet.ml/site/>`__.
    """






class AzureMLCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `AzureML
    <https://pypi.org/project/azureml-sdk/>`__.
    """





class MLflowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `MLflow <https://www.mlflow.org/>`__.
    """

    MAX_LOG_SIZE = 100





