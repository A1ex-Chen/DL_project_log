from typing import Dict, Any, List, Tuple

from PIL import Image
from mmengine import DATASETS, TRANSFORMS, METRICS, FUNCTIONS, Registry

from ..conversation import Conversation

IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
# processor
BOXES_PROCESSOR = Registry('Processor for Boxes')


# only for static type checking
class BaseConvProcessFunc:


class BaseTargetProcessFunc:


class BaseTextProcessFunc:


class BaseImageProcessFunc:


__all__ = [
    'IMAGE_PLACEHOLDER', 'BOXES_PLACEHOLDER', 'EXPR_PLACEHOLDER', 'OBJS_PLACEHOLDER', 'QUESTION_PLACEHOLDER', 'POINTS_PLACEHOLDER',
    'FUNCTIONS',
    'DATASETS',
    'TRANSFORMS',
    'METRICS',
    'BOXES_PROCESSOR',
    'BaseConvProcessFunc', 'BaseTargetProcessFunc', 'BaseTextProcessFunc', 'BaseImageProcessFunc',
]