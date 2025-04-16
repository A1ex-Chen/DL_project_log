from copy import deepcopy
from typing import Any, Mapping

from accelerate.logging import get_logger
from models.audio_consistency_model import AudioLCM
from tools.train_utils import do_ema_update

logger = get_logger(__name__, log_level="INFO")


class AudioLCM_FTVAE(AudioLCM):





