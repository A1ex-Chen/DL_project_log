#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
import torch
from ..mplug_owl import *

                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading mPLUG-Owl2 from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if 'mplug_owl2_1' in model_name.lower():
                model = MPLUGOwl2QWenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            if 'mplug_owl2_1' in model_name.lower():
                model = MPLUGOwl2QWenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                model = MPLUGOwl2LlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)


    vision_tower = model.get_model().vision_model
    # vision_tower.to(device=device, dtype=torch.float16)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len