import os

from transformers import CLIPTokenizer, CLIPTokenizerFast
from transformers import AutoTokenizer

from .registry import lang_encoders
from .registry import is_lang_encoder



