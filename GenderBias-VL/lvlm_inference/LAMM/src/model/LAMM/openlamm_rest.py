import logging
import io
import numpy as np
import os
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image, ImageFile
import requests
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from transformers import LlamaForCausalLM, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList

from .CLIP import load as load_clip
import model.LAMM.conversations as conversations
from .modeling_llama import LlamaForCausalLM
from .utils.pcl_utils import MEAN_COLOR_RGB, random_sampling
from .utils.data import transform_vision_data

# load optional 3d encoder
try:
    LOAD_EPCL_EXT = True
    from .EPCL import build_epcl_encoder
except ImportError as e:
    LOAD_EPCL_EXT = False
    logging.warning(f'{e.msg}. Please refer to README.md to install optional extension for 3D environment if required.')

# load optional lightllm
try:
    LOAD_LIGHTLLM_EXT = True
    from .modeling_lightllm import LlamaLightForCausalLM
except ImportError as e:
    LOAD_LIGHTLLM_EXT = False
    logging.warning(f'{e.msg}. Please refer to README.md to install optional LightLLM extension if required.')

ImageFile.LOAD_TRUNCATED_IMAGES = True


VISION_TAGS = {
    "pos": {"image": "<image>", "pcl": "<pcl>"},
    "sov": {"image": "<Img>", "pcl": "<Pcl>"},
    "eov": {"image": "</Img>", "pcl": "</Pcl>"},
}


class LAMMStoppingCriteria(StoppingCriteria):
    def __init__(self, stops, input_ids, device):
        """intialize stopping criteria

        :param list stops: list of stop tokens
        :param list input_ids: input ids
        """
        super().__init__()
        self.stops = [torch.tensor(stop).to(device) for stop in stops]
        self.stop_flag = [0] * input_ids.shape[0]

    def check_stop(self, input_ids):
        """check whether to stop generation

        :param list input_ids: input token ids
        :return bool: stop or not
        """
        for stop in self.stops:
            if torch.all((stop == input_ids[-len(stop):])).item():
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """call function of stop creteria

        :param torch.LongTensor output_ids: output token ids
        :return bool: stop or not
        """
        flag = 1
        for id, output_id in enumerate(output_ids):
            if self.stop_flag[id] == 1:
                continue
            if self.check_stop(output_id):
                self.stop_flag[id] = 1
            else:
                flag = 0
        if flag == 1:
            return True
        return False










def build_one_instance(tokenizer, conversation, vision_type="image", template=conversations.default_conversation):
    """build one instance for training; text part

    :param class tokenizer: text tokenizer
    :param list conversation: list of conversation
    :param str vision_type: type of vision data, defaults to 'image'
    :raises Exception: Exception if wrong role included
    :return list: conversation text list, input token ids, target token ids
    """
    pos = VISION_TAGS["pos"][vision_type]
    eov = VISION_TAGS["eov"][vision_type]
    text_list = []
    turn_num = len(conversation)
    input_ids, target_ids = [], []
    for i in range(turn_num):
        turn = conversation[i]
        role = turn["from"]
        if i == 0:  # the first human turn
            assert role == "human"
            turn["value"] = (
                turn["value"].replace(f"{pos}\n", "").replace(f"\n{pos}", "")
            )
            text = f"{eov} " + turn["value"] + "\n{} {}:".format(template.sep, template.roles[1])
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(
                one_input_id
            )  # do not perform loss regression on human prompt
        else:
            if role == "human":
                # text = "{}: ".format(template.roles[0]) + turn["value"] + "\n### {}:".format(template.roles[1])
                text = "{}: {}\n{} {}:".format(template.roles[0], turn["value"], template.sep, template.roles[1])
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += [-100] * len(one_input_id)
            elif role == "gpt":
                text = turn["value"] + "\n{}".format(template.sep2 if (template.sep2 is not None) else template.sep)
                one_input_id = tokenizer(text, add_special_tokens=False).input_ids
                input_ids += one_input_id
                target_ids += one_input_id
            else:
                raise Exception(f"{role} is a Wrong Role!!!")
        text_list.append(text)
        assert len(input_ids) == len(target_ids)
    return text_list, input_ids, target_ids


def process_batch_instance(
    tokenizer, batch_of_conversations, max_tgt_len, vision_type="image", template=conversations.default_conversation
):
    """build one batch of instance for training

    :param class tokenizer: text tokenizer
    :param list batch_of_conversations: batch of conversations
    :param int max_tgt_len: max token length of after vision tokens
    :param str vision_type: type of vision data, defaults to 'image'
    :return list: input token ids, target token ids, attention mask
    """
    batch_input_ids, batch_target_ids = [], []
    for conversation in batch_of_conversations:
        _, one_input_ids, one_target_ids = build_one_instance(
            tokenizer, conversation, vision_type=vision_type, template=template
        )
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    target_ids = rnn.pad_sequence(
        batch_target_ids, batch_first=True, padding_value=-100
    )
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def make_prompt_start(use_system=False, vision_type="image", task_type="normal", template=conversations.default_conversation):
    """make starting prompt

    :param bool use_system: whether to use system message, defaults to False
    :param str vision_type: type of visio data, defaults to 'image'
    :param str task_type: task type of current sample, defaults to 'normal'
    :return str: resulting starting prompt
    """
    # PROMPT_START = f'### Human: {VISION_TAGS["sov"][vision_type]}'
    PROMPT_START = f'{template.sep} {template.roles[0]}: {VISION_TAGS["sov"][vision_type]}'
    if use_system:
        if task_type == "normal":
            # print(template.system)
            return f"{template.system}\n\n" + PROMPT_START
        else:
            if template.sys_temp is None:
                return [
                    f"{conversations.conversation_dict[task]}\n\n" + PROMPT_START
                    for task in task_type
                ]
            else:
                # print(template.sys_temp.format(system_message=conversations.conversation_dict[task_type[0]]))
                return [template.sys_temp.format(system_message=conversations.conversation_dict[task]) + PROMPT_START for task in task_type]
    else:
        return PROMPT_START


class LAMMPEFTModel(nn.Module):
    """LoRA for LAMM model"""

        
    



    















class LAMMSFTModel(LAMMPEFTModel):
    """SFT for LAMM model"""

        # self.llama_model.print_trainable_parameters()


