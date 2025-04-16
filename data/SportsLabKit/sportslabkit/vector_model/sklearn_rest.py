from typing import Any

from joblib import load
from sklearn.pipeline import Pipeline

from sportslabkit.types import Vector
from sportslabkit.utils import fetch_or_cache_model
from sportslabkit.vector_model.base import BaseVectorModel


class SklearnVectorModel(BaseVectorModel):
    """
    A specialized subclass of BaseVectorModel for scikit-learn pipelines.

    This class is designed to facilitate the use of scikit-learn pipelines as vector-based models
    within the SportsLabKit ecosystem. It overrides the abstract methods from BaseVectorModel
    to provide implementations tailored for scikit-learn pipelines.

    Attributes:
        model (Pipeline | None): The loaded scikit-learn pipeline model. None if the model is not loaded.
    """


