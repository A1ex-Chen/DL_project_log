# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model DeBERTa."""

import os
import pathlib
import random
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple
from zipfile import ZipFile

import tqdm

import requests

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


try:
    import regex as re
except ImportError:
    raise ImportError("Please install regex with: pip install regex")


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "bpe_encoder.bin"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/bpe_encoder.bin",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/bpe_encoder.bin",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

__all__ = ["DebertaTokenizer"]


@lru_cache()




class Encoder:
    def __init__(self, encoder, bpe_merges, errors="replace"):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip([tuple(k) for k in bpe_merges], range(len(bpe_merges))))
        self.cache = {}
        self.random = random.Random(0)

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def split_to_words(self, text):
        return list(re.findall(self.pat, text))

    def encode(self, text):
        bpe_tokens = []
        for token in self.split_to_words(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text


















def get_encoder(encoder, vocab):
    return Encoder(
        encoder=encoder,
        bpe_merges=vocab,
    )


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def download_asset(name, tag=None, no_cache=False, cache_dir=None):
    _tag = tag
    if _tag is None:
        _tag = "latest"
    if not cache_dir:
        cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
    os.makedirs(cache_dir, exist_ok=True)
    output = os.path.join(cache_dir, name)
    if os.path.exists(output) and (not no_cache):
        return output

    repo = "https://api.github.com/repos/microsoft/DeBERTa/releases"
    releases = requests.get(repo).json()
    if tag and tag != "latest":
        release = [r for r in releases if r["name"].lower() == tag.lower()]
        if len(release) != 1:
            raise Exception(f"{tag} can't be found in the repository.")
    else:
        release = releases[0]
    asset = [s for s in release["assets"] if s["name"].lower() == name.lower()]
    if len(asset) != 1:
        raise Exception(f"{name} can't be found in the release.")
    url = asset[0]["url"]
    headers = {}
    headers["Accept"] = "application/octet-stream"
    resp = requests.get(url, stream=True, headers=headers)
    if resp.status_code != 200:
        raise Exception(f"Request for {url} return {resp.status_code}, {resp.text}")
    try:
        with open(output, "wb") as fs:
            progress = tqdm(
                total=int(resp.headers["Content-Length"]) if "Content-Length" in resp.headers else -1,
                ncols=80,
                desc=f"Downloading {name}",
            )
            for c in resp.iter_content(chunk_size=1024 * 1024):
                fs.write(c)
            progress.update(len(c))
            progress.close()
    except Exception:
        os.remove(output)
        raise

    return output


def load_vocab(name=None, tag=None, no_cache=False, cache_dir=None):
    import torch

    if name is None:
        name = "bpe_encoder"

    model_path = name
    if model_path and (not os.path.exists(model_path)) and not (("/" in model_path) or ("\\" in model_path)):
        _tag = tag
        if _tag is None:
            _tag = "latest"
        if not cache_dir:
            cache_dir = os.path.join(pathlib.Path.home(), f".~DeBERTa/assets/{_tag}/")
        os.makedirs(cache_dir, exist_ok=True)
        out_dir = os.path.join(cache_dir, name)
        model_path = os.path.join(out_dir, "bpe_encoder.bin")
        if (not os.path.exists(model_path)) or no_cache:
            asset = download_asset(name + ".zip", tag=tag, no_cache=no_cache, cache_dir=cache_dir)
            with ZipFile(asset, "r") as zipf:
                for zip_info in zipf.infolist():
                    if zip_info.filename[-1] == "/":
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zipf.extract(zip_info, out_dir)
    elif not model_path:
        return None, None

    encoder_state = torch.load(model_path)
    return encoder_state


class GPT2Tokenizer(object):
    """
    A wrapper of GPT2 tokenizer with similar interface as BERT tokenizer

    Args:
        vocab_file (:obj:`str`, optional):
            The local path of vocabulary package or the release name of vocabulary in `DeBERTa GitHub releases
            <https://github.com/microsoft/DeBERTa/releases>`_, e.g. "bpe_encoder", default: `None`.

            If it's `None`, then it will download the vocabulary in the latest release from GitHub. The vocabulary file
            is a state dictionary with three items, "dict_map", "vocab", "encoder" which correspond to three files used
            in `RoBERTa`, i.e. `dict.txt`, `vocab.txt` and `encoder.json`. The difference between our wrapped GPT2
            tokenizer and RoBERTa wrapped tokenizer are,

            - Special tokens, unlike `RoBERTa` which use `<s>`, `</s>` as the `start` token and `end` token of a
              sentence. We use `[CLS]` and `[SEP]` as the `start` and `end` token of input sentence which is the same
              as `BERT`.

            - We remapped the token ids in our dictionary with regarding to the new special tokens, `[PAD]` => 0,
              `[CLS]` => 1, `[SEP]` => 2, `[UNK]` => 3, `[MASK]` => 50264

        special_tokens (:obj:`list`, optional):
            List of special tokens to be added to the end of the vocabulary.
    """
















class DebertaTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a DeBERTa tokenizer, which runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


    @property

    @property









