import math
import logging
from datasets import load_dataset

from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    is_torch_tpu_available,
    set_seed,
)
from mlm_util import ModelArguments, DataTrainingArguments, ExtendedTrainingArguments
from util import is_main_process, init_logger, init_output_dir
from util import format_args, preprocess_logits_for_metrics
from mlm_util import get_metric_function, get_preprocess_function
from mlm_util import get_token2id_mapping, MyDataCollator

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)




if __name__ == "__main__":
    main()