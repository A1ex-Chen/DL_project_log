import os

from structlog import get_logger

from modelkit.assets.manager import AssetsManager
from modelkit.assets.settings import AssetSpec
from modelkit.core.library import download_assets
from modelkit.core.models.tensorflow_model import TensorflowModel

logger = get_logger(__name__)



