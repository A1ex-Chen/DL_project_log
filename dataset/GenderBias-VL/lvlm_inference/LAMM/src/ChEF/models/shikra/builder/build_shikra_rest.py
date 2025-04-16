from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn

from ..shikra import ShikraLlamaForCausalLM, ShikraConfig

PREPROCESSOR = Dict[str, Any]

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"





                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg