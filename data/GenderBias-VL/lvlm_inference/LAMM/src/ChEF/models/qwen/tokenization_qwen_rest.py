# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenization classes for QWen."""

import base64
import logging
import os
import requests
import unicodedata
from typing import Collection, Dict, List, Set, Tuple, Union, Any, Callable, Optional

import tiktoken
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from transformers import PreTrainedTokenizer, AddedToken
from transformers.utils import try_to_load_from_cache

import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "qwen.tiktoken", "ttf": "SimSun.ttf"}
FONT_PATH = try_to_load_from_cache("Qwen/Qwen-VL-Chat", "SimSun.ttf")
if FONT_PATH is None:
    if not os.path.exists("SimSun.ttf"):
        ttf = requests.get("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/SimSun.ttf")
        open("SimSun.ttf", "wb").write(ttf.content)
    FONT_PATH = "SimSun.ttf"

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
# as the default behavior is changed to allow special tokens in
# regular texts, the surface forms of special tokens need to be
# as different as possible to minimize the impact
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
    ENDOFTEXT,
    IMSTART,
    IMEND,
) + EXTRAS
IMG_TOKEN_SPAN = 256





class QWenTokenizer(PreTrainedTokenizer):
    """QWen tokenizer."""

    vocab_files_names = VOCAB_FILES_NAMES












    @property











import colorsys
import logging
import math
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import random

logger = logging.getLogger(__name__)


class VisImage:






class Visualizer:




        return _replace_closed_tag(tokens, self.image_start_tag, self.image_end_tag, _encode_imgurl)

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    def _tokenize(self, text: str, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        errors: str = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]


        token_ids = _replace_closed_tag(token_ids, self.img_start_id, self.img_end_id, _decode_imgurl)

        if skip_special_tokens:
            if kwargs.get('keep_image_special', False):
                token_ids = [i for i in token_ids if i < self.eod_id 
                    or i in self.image_special_tokens]
            else:
                token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=errors or self.errors)

    def to_list_format(self, text: str):
        text = unicodedata.normalize("NFC", text)
        token_ids = self.tokenizer.encode(
            text, allowed_special=set(self.IMAGE_ST + (ENDOFTEXT,)))


        return _replace_closed_tag(
            token_ids,
            (self.img_start_id, self.ref_start_id, self.box_start_id, self.quad_start_id),
            (self.img_end_id, self.ref_end_id, self.box_end_id, self.quad_end_id),
            _encode_vl_info,
            _encode_vl_info,
        )

    def from_list_format(self, list_format: List[Dict]):
        text = ''
        num_images = 0
        for ele in list_format:
            if 'image' in ele:
                num_images += 1
                text += f'Picture {num_images}: '
                text += self.image_start_tag + ele['image'] + self.image_end_tag
                text += '\n'
            elif 'text' in ele:
                text += ele['text']
            elif 'box' in ele:
                if 'ref' in ele:
                    text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
                for box in ele['box']:
                    text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0], box[1], box[2], box[3]) + self.box_end_tag
            else:
                raise ValueError("Unsupport element: " + str(ele))
        return text

    def _fetch_latest_picture(self, response, history):
        if history is None:
            history = []
        _history = history + [(response, None)]
        for q, r in _history[::-1]:
            for ele in self.to_list_format(q)[::-1]:
                if 'image' in ele:
                    return ele['image']
        return None

    def _fetch_all_box_with_ref(self, text):
        list_format = self.to_list_format(text)
        output = []
        for i, ele in enumerate(list_format):
            if 'box' in ele:
                bbox = tuple(map(int, ele['box'].replace('(', '').replace(')', '').split(',')))
                assert len(bbox) == 4
                output.append({'box': bbox})
                if i > 0 and 'ref' in list_format[i-1]:
                    output[-1]['ref'] = list_format[i-1]['ref'].strip()
        return output

    def draw_bbox_on_latest_picture(
        self,
        response,
        history=None,
    ) -> Optional[Image.Image]:
        image = self._fetch_latest_picture(response, history)
        if image is None:
            return None
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            h, w = image.height, image.width
        else:
            image = np.asarray(Image.open(image).convert("RGB"))
            h, w = image.shape[0], image.shape[1]
        visualizer = Visualizer(image)

        boxes = self._fetch_all_box_with_ref(response)
        if not boxes:
            return None
        color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()]) # init color
        for box in boxes:
            if 'ref' in box: # random new color for new refexps
                color = random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()])
            x1, y1, x2, y2 = box['box']
            x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
            visualizer.draw_box((x1, y1, x2, y2), alpha=1, edge_color=color)
            if 'ref' in box:
                visualizer.draw_text(box['ref'], (x1, y1), color=color, horizontal_alignment="left")
        return visualizer.output


import colorsys
import logging
import math
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import random

logger = logging.getLogger(__name__)


class VisImage:
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    def __init__(self, img_rgb, metadata=None, scale=1.0):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.font_path = FONT_PATH
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 14
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 30, 15 // scale
        )

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            fontproperties=FontProperties(fname=self.font_path),
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def get_output(self):
        
        return self.output