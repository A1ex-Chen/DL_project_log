import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile

import numpy as np
import torch
from tqdm import tqdm

from transformers import MarianConfig, MarianMTModel, MarianTokenizer
from transformers.hf_api import HfApi


















CONFIG_KEY = "special:model.yml"






# Group Names Logic: change long opus model names to something shorter, like opus-mt-en-ROMANCE
ROM_GROUP = (
    "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO+es_EC+es_ES+es_GT"
    "+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR+pt_PT+gl+lad+an+mwl+it+it_IT+co"
    "+nap+scn+vec+sc+ro+la"
)
GROUPS = [
    ("cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh", "ZH"),
    (ROM_GROUP, "ROMANCE"),
    ("de+nl+fy+af+da+fo+is+no+nb+nn+sv", "NORTH_EU"),
    ("da+fo+is+no+nb+nn+sv", "SCANDINAVIA"),
    ("se+sma+smj+smn+sms", "SAMI"),
    ("nb_NO+nb+nn_NO+nn+nog+no_nb+no", "NORWAY"),
    ("ga+cy+br+gd+kw+gv", "CELTIC"),  # https://en.wikipedia.org/wiki/Insular_Celtic_languages
]
GROUP_TO_OPUS_NAME = {
    "opus-mt-ZH-de": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-de",
    "opus-mt-ZH-fi": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-fi",
    "opus-mt-ZH-sv": "cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh-sv",
    "opus-mt-SCANDINAVIA-SCANDINAVIA": "da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv",
    "opus-mt-NORTH_EU-NORTH_EU": "de+nl+fy+af+da+fo+is+no+nb+nn+sv-de+nl+fy+af+da+fo+is+no+nb+nn+sv",
    "opus-mt-de-ZH": "de-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-en_el_es_fi-en_el_es_fi": "en+el+es+fi-en+el+es+fi",
    "opus-mt-en-ROMANCE": "en-fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
    "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
    "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la",
    "opus-mt-en-CELTIC": "en-ga+cy+br+gd+kw+gv",
    "opus-mt-es-NORWAY": "es-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-fi_nb_no_nn_ru_sv_en-SAMI": "fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms",
    "opus-mt-fi-ZH": "fi-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-fi-NORWAY": "fi-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
    "opus-mt-ROMANCE-en": "fr+fr_BE+fr_CA+fr_FR+wa+frp+oc+ca+rm+lld+fur+lij+lmo+es+es_AR+es_CL+es_CO+es_CR+es_DO"
    "+es_EC+es_ES+es_GT+es_HN+es_MX+es_NI+es_PA+es_PE+es_PR+es_SV+es_UY+es_VE+pt+pt_br+pt_BR"
    "+pt_PT+gl+lad+an+mwl+it+it_IT+co+nap+scn+vec+sc+ro+la-en",
    "opus-mt-CELTIC-en": "ga+cy+br+gd+kw+gv-en",
    "opus-mt-sv-ZH": "sv-cmn+cn+yue+ze_zh+zh_cn+zh_CN+zh_HK+zh_tw+zh_TW+zh_yue+zhs+zht+zh",
    "opus-mt-sv-NORWAY": "sv-nb_NO+nb+nn_NO+nn+nog+no_nb+no",
}
OPUS_GITHUB_URL = "https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/"
ORG_NAME = "Helsinki-NLP/"








# docstyle-ignore
FRONT_MATTER_TEMPLATE = """---
language:
{}
tags:
- translation

license: apache-2.0
---
"""
DEFAULT_REPO = "Tatoeba-Challenge"
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_REPO, "models")




























BIAS_KEY = "decoder_ff_logit_out_b"
BART_CONVERTER = {  # for each encoder and decoder layer
    "self_Wq": "self_attn.q_proj.weight",
    "self_Wk": "self_attn.k_proj.weight",
    "self_Wv": "self_attn.v_proj.weight",
    "self_Wo": "self_attn.out_proj.weight",
    "self_bq": "self_attn.q_proj.bias",
    "self_bk": "self_attn.k_proj.bias",
    "self_bv": "self_attn.v_proj.bias",
    "self_bo": "self_attn.out_proj.bias",
    "self_Wo_ln_scale": "self_attn_layer_norm.weight",
    "self_Wo_ln_bias": "self_attn_layer_norm.bias",
    "ffn_W1": "fc1.weight",
    "ffn_b1": "fc1.bias",
    "ffn_W2": "fc2.weight",
    "ffn_b2": "fc2.bias",
    "ffn_ffn_ln_scale": "final_layer_norm.weight",
    "ffn_ffn_ln_bias": "final_layer_norm.bias",
    # Decoder Cross Attention
    "context_Wk": "encoder_attn.k_proj.weight",
    "context_Wo": "encoder_attn.out_proj.weight",
    "context_Wq": "encoder_attn.q_proj.weight",
    "context_Wv": "encoder_attn.v_proj.weight",
    "context_bk": "encoder_attn.k_proj.bias",
    "context_bo": "encoder_attn.out_proj.bias",
    "context_bq": "encoder_attn.q_proj.bias",
    "context_bv": "encoder_attn.v_proj.bias",
    "context_Wo_ln_scale": "encoder_attn_layer_norm.weight",
    "context_Wo_ln_bias": "encoder_attn_layer_norm.bias",
}


class OpusState:
    def __init__(self, source_dir):
        npz_path = find_model_file(source_dir)
        self.state_dict = np.load(npz_path)
        cfg = load_config_from_state_dict(self.state_dict)
        assert cfg["dim-vocabs"][0] == cfg["dim-vocabs"][1]
        assert "Wpos" not in self.state_dict, "Wpos key in state dictionary"
        self.state_dict = dict(self.state_dict)
        self.wemb, self.final_bias = add_emb_entries(self.state_dict["Wemb"], self.state_dict[BIAS_KEY], 1)
        self.pad_token_id = self.wemb.shape[0] - 1
        cfg["vocab_size"] = self.pad_token_id + 1
        # self.state_dict['Wemb'].sha
        self.state_keys = list(self.state_dict.keys())
        assert "Wtype" not in self.state_dict, "Wtype key in state dictionary"
        self._check_layer_entries()
        self.source_dir = source_dir
        self.cfg = cfg
        hidden_size, intermediate_shape = self.state_dict["encoder_l1_ffn_W1"].shape
        assert (
            hidden_size == cfg["dim-emb"] == 512
        ), f"Hidden size {hidden_size} and configured size {cfg['dim_emb']} mismatched or not 512"

        # Process decoder.yml
        decoder_yml = cast_marian_config(load_yaml(source_dir / "decoder.yml"))
        check_marian_cfg_assumptions(cfg)
        self.hf_config = MarianConfig(
            vocab_size=cfg["vocab_size"],
            decoder_layers=cfg["dec-depth"],
            encoder_layers=cfg["enc-depth"],
            decoder_attention_heads=cfg["transformer-heads"],
            encoder_attention_heads=cfg["transformer-heads"],
            decoder_ffn_dim=cfg["transformer-dim-ffn"],
            encoder_ffn_dim=cfg["transformer-dim-ffn"],
            d_model=cfg["dim-emb"],
            activation_function=cfg["transformer-aan-activation"],
            pad_token_id=self.pad_token_id,
            eos_token_id=0,
            bos_token_id=0,
            max_position_embeddings=cfg["dim-emb"],
            scale_embedding=True,
            normalize_embedding="n" in cfg["transformer-preprocess"],
            static_position_embeddings=not cfg["transformer-train-position-embeddings"],
            dropout=0.1,  # see opus-mt-train repo/transformer-dropout param.
            # default: add_final_layer_norm=False,
            num_beams=decoder_yml["beam-size"],
            decoder_start_token_id=self.pad_token_id,
            bad_words_ids=[[self.pad_token_id]],
            max_length=512,
        )

    def _check_layer_entries(self):
        self.encoder_l1 = self.sub_keys("encoder_l1")
        self.decoder_l1 = self.sub_keys("decoder_l1")
        self.decoder_l2 = self.sub_keys("decoder_l2")
        if len(self.encoder_l1) != 16:
            warnings.warn(f"Expected 16 keys for each encoder layer, got {len(self.encoder_l1)}")
        if len(self.decoder_l1) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")
        if len(self.decoder_l2) != 26:
            warnings.warn(f"Expected 26 keys for each decoder layer, got {len(self.decoder_l1)}")

    @property
    def extra_keys(self):
        extra = []
        for k in self.state_keys:
            if (
                k.startswith("encoder_l")
                or k.startswith("decoder_l")
                or k in [CONFIG_KEY, "Wemb", "Wpos", "decoder_ff_logit_out_b"]
            ):
                continue
            else:
                extra.append(k)
        return extra

    def sub_keys(self, layer_prefix):
        return [remove_prefix(k, layer_prefix) for k in self.state_dict if k.startswith(layer_prefix)]

    def load_marian_model(self) -> MarianMTModel:
        state_dict, cfg = self.state_dict, self.hf_config

        assert cfg.static_position_embeddings, "config.static_position_embeddings should be True"
        model = MarianMTModel(cfg)

        assert "hidden_size" not in cfg.to_dict()
        load_layers_(
            model.model.encoder.layers,
            state_dict,
            BART_CONVERTER,
        )
        load_layers_(model.model.decoder.layers, state_dict, BART_CONVERTER, is_decoder=True)

        # handle tensors not associated with layers
        wemb_tensor = torch.nn.Parameter(torch.FloatTensor(self.wemb))
        bias_tensor = torch.nn.Parameter(torch.FloatTensor(self.final_bias))
        model.model.shared.weight = wemb_tensor
        model.model.encoder.embed_tokens = model.model.decoder.embed_tokens = model.model.shared

        model.final_logits_bias = bias_tensor

        if "Wpos" in state_dict:
            print("Unexpected: got Wpos")
            wpos_tensor = torch.tensor(state_dict["Wpos"])
            model.model.encoder.embed_positions.weight = wpos_tensor
            model.model.decoder.embed_positions.weight = wpos_tensor

        if cfg.normalize_embedding:
            assert "encoder_emb_ln_scale_pre" in state_dict
            raise NotImplementedError("Need to convert layernorm_embedding")

        assert not self.extra_keys, f"Failed to convert {self.extra_keys}"
        assert (
            model.model.shared.padding_idx == self.pad_token_id
        ), f"Padding tokens {model.model.shared.padding_idx} and {self.pad_token_id} mismatched"
        return model












    @property




def download_and_unzip(url, dest_dir):
    try:
        import wget
    except ImportError:
        raise ImportError("you must pip install wget")

    filename = wget.download(url)
    unzip(filename, dest_dir)
    os.remove(filename)


def convert(source_dir: Path, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)

    add_special_tokens_to_vocab(source_dir)
    tokenizer = MarianTokenizer.from_pretrained(str(source_dir))
    tokenizer.save_pretrained(dest_dir)

    opus_state = OpusState(source_dir)
    assert opus_state.cfg["vocab_size"] == len(
        tokenizer.encoder
    ), f"Original vocab size {opus_state.cfg['vocab_size']} and new vocab size {len(tokenizer.encoder)} mismatched"
    # save_json(opus_state.cfg, dest_dir / "marian_original_config.json")
    # ^^ Uncomment to save human readable marian config for debugging

    model = opus_state.load_marian_model()
    model = model.half()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)  # sanity check


def load_yaml(path):
    import yaml

    with open(path) as f:
        return yaml.load(f, Loader=yaml.BaseLoader)


def save_json(content: Union[Dict, List], path: str) -> None:
    with open(path, "w") as f:
        json.dump(content, f)


def unzip(zip_path: str, dest_dir: str) -> None:
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(dest_dir)


if __name__ == "__main__":
    """
    Tatoeba conversion instructions in scripts/tatoeba/README.md
    """
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--src", type=str, help="path to marian model sub dir", default="en-de")
    parser.add_argument("--dest", type=str, default=None, help="Path to the output PyTorch model.")
    args = parser.parse_args()

    source_dir = Path(args.src)
    assert source_dir.exists(), f"Source directory {source_dir} not found"
    dest_dir = f"converted-{source_dir.name}" if args.dest is None else args.dest
    convert(source_dir, dest_dir)