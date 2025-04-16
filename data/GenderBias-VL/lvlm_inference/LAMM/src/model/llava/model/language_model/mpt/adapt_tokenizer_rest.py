from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
NUM_SENTINEL_TOKENS: int = 100


class AutoTokenizerForMOD(AutoTokenizer):
    """AutoTokenizer + Adaptation for MOD.

    A simple wrapper around AutoTokenizer to make instantiating
    an MOD-adapted tokenizer a bit easier.

    MOD-adapted tokenizers have sentinel tokens (e.g., <extra_id_0>),
    a padding token, and a property to get the token ids of the
    sentinel tokens.
    """

    @classmethod