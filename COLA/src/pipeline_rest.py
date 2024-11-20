"""
pipeline.py
"""
import copy
import itertools, re

import numpy as np
import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
import allennlp_models.pretrained
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


class AllenSRLWrapper:
    PERTURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    EMPTY_TOK = "[EMPTY]"
    ANSWER_TOK = "[ANSWER]"





    @staticmethod




"""
temporal predictor

"""


class TempPredictor:






"""
polyjuice intervention
"""


class PJGenerator:
    PERTURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    EMPTY_TOK = "[EMPTY]"
    ANSWER_TOK = "[ANSWER]"





        srl_list = self.predict_batch_json(text_batch)
        res_list = []
        for srl in srl_list:
            tokens = srl['words']
            verbs = srl['verbs']
            targets = ['ARG0', 'V', 'ARG1']
            blanks = []
            for v in verbs:
                tags = v['tags']
                for tgt in targets:
                    if f"B-{tgt}" not in tags:
                        continue
                    blk_start = tags.index(f"B-{tgt}")
                    blk_end = blk_start + 1 if f"I-{tgt}" not in tags else len(tags) - tags[::-1].index(f"I-{tgt}")
                    sent = ' '.join(tokens[:blk_start]) + " [BLANK] " + ' '.join(tokens[blk_end:])
                    blanks.append(proper_whitespaces(sent))
            blanks = list(set(blanks))
            if not blanks:
                blanks = ["[BLANK]"]
            res_list.append(blanks)
        return res_list

    def get_prompts(self, text, ctrl_codes, blanked_sents, is_complete_blank=True):
        prompts = []
        for tag, bt in itertools.product(ctrl_codes, blanked_sents):
            sep_tok = self.SEP_TOK if bt and is_complete_blank else ""
            prompts.append(f"{text.strip()} {self.PERTURB_TOK} [{tag}] {bt.strip()} {sep_tok}".strip())
        return prompts


"""
temporal predictor

"""


class TempPredictor:
    def __init__(self, model, tokenizer, device):
        self._model = model
        self._model.to(device)
        self._device = self._model.device
        self._model.eval()
        self._tokenizer = tokenizer
        self._mtoken = self._tokenizer.mask_token

        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

    def _extract_token_prob(self, arr, token, crop=1):
        for it in arr:
            if len(it["token_str"]) >= crop and (token == it["token_str"][crop:]):
                return it["score"]
        return 0.

    def batch_predict(self, input_dataset, batch_size, top_k):
        device = self._device
        data_collator = DataCollatorWithPadding(self._tokenizer, padding=True)
        train_dataloader = DataLoader(
            input_dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size)

        topk_value_list, topk_indices_list = [], []
        with torch.no_grad():
            for data_batch in tqdm(train_dataloader, "predicting"):
                cur_batch_size = len(data_batch["input_ids"])
                data_batch.to(device)
                mask_token_index = torch.where(data_batch["input_ids"] == self._tokenizer.mask_token_id)[1]
                token_logits = self._model(**data_batch).logits
                mask_token_logits = token_logits[torch.arange(cur_batch_size), mask_token_index, :]

                mask_token_logits = F.softmax(mask_token_logits, dim=1)
                top_k_values, top_k_indices = torch.topk(mask_token_logits, k=top_k, dim=1)

                topk_value_list.append(top_k_values.cpu().numpy())
                topk_indices_list.append(top_k_indices.cpu().numpy())

        topk_value_list = np.concatenate(topk_value_list)
        topk_indices_list = np.concatenate(topk_indices_list)
        return topk_value_list, topk_indices_list

    def decoding_logits(self, preds_tuple):
        topk_values, topk_indices = preds_tuple
        pred_list = []
        for value_list, indices_list in tqdm(zip(topk_values, topk_indices), "decoding tokens",
                                             total=len(topk_indices)):
            cur_pred_list = []
            for v, i in zip(value_list, indices_list):
                cur_pred_list.append({"token_str": self._tokenizer.decode([i]), "score": v})
            pred_list.append(cur_pred_list)
        return pred_list

    def postprocess_prob(self, fwd_tuple, bwd_tuple, device, crop=1):
        fwd_pred_list = self.decoding_logits(fwd_tuple)
        bwd_pred_list = self.decoding_logits(bwd_tuple)

        forward_before = torch.tensor([self._extract_token_prob(pred, "before", crop=crop) for pred in fwd_pred_list])
        backward_before = torch.tensor([self._extract_token_prob(pred, "after", crop=crop) for pred in bwd_pred_list])

        forward_after = torch.tensor([self._extract_token_prob(pred, "after", crop=crop) for pred in fwd_pred_list])
        backward_after = torch.tensor([self._extract_token_prob(pred, "before", crop=crop) for pred in bwd_pred_list])

        forward_before, backward_before = forward_before.to(device), backward_before.to(device)
        forward_after, backward_after = forward_after.to(device), backward_after.to(device)

        avg_before = (forward_before + backward_before) / 2
        avg_after = (forward_after + backward_after) / 2

        avg = torch.stack([avg_before, avg_after], dim=-1).cpu().numpy()
        return avg


"""
polyjuice intervention
"""


class PJGenerator:
    PERTURB_TOK = "<|perturb|>"
    BLANK_TOK = "[BLANK]"
    SEP_TOK = "[SEP]"
    EMPTY_TOK = "[EMPTY]"
    ANSWER_TOK = "[ANSWER]"

    def __init__(self, model_path=None,
                 device=None):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        generator = transformers.pipeline("text-generation",
                                          model=AutoModelForCausalLM.from_pretrained(model_path),
                                          tokenizer=tokenizer,
                                          framework="pt", device=device)
        self.tokenizer = tokenizer
        self.generator = generator

    def generate_on_prompts(self, generator, prompts, **kwargs):




        preds_list = batched_generate(generator, prompts, **kwargs)

        if len(prompts) == 1:
            preds_list = [preds_list]

        preds_list_cleaned = []
        for prompt, preds in zip(prompts, preds_list):
            prev_list = set()
            for s in preds:
                total_sequence = s["generated_text"].split(self.PERTURB_TOK)[-1]
                normalized, _ = remove_blanks(total_sequence)
                input_ctrl_code, normalized = split_ctrl_code(normalized)
                prev_list.add((input_ctrl_code, normalized))
            preds_list_cleaned.append(list(prev_list))
        return preds_list_cleaned

    def agg_generations(self, gen):
        agg = {}
        for lists in gen:
            for (ctrl, sent) in lists:
                if ctrl not in agg:
                    agg[ctrl] = []
                agg[ctrl].append(sent)
        return agg

    def __call__(self, prompts,
                 aggregation=True,
                 **kwargs):
        generations = self.generate_on_prompts(self.generator, prompts, **kwargs)
        if aggregation:
            generations = self.agg_generations(generations)
        generations = list(itertools.chain(*[ints for _, ints in generations.items()]))
        generations = [g.strip() for g in generations]
        return generations