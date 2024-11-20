import os
import gzip
from sre_parse import SPECIAL_CHARS
import numpy as np
from random import Random
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union
import collections
from infinibatch import iterators

EOD_SYMBOL = "</doc>"
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
EOC_SYMBOL = "</chunk>"
EOL_SYMBOL = "</line>"

GRD_SYMBOL="<grounding>"
BOP_SYMBOL="<phrase>"
EOP_SYMBOL="</phrase>"
BOO_SYMBOL="<object>"
EOO_SYMBOL="</object>"
DOM_SYMBOL="</delimiter_of_multi_objects/>"

SPECIAL_SYMBOLS = [EOD_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, EOC_SYMBOL, EOL_SYMBOL]



    return _apply(sample)

class NativeCheckpointableIterator(iterators.CheckpointableIterator):



    


class WeightIterator(object):
        
    

        
    


class ConcatIterator(iterators.CheckpointableIterator):
    """
    Concat items from all given iterators.
    """



    


class MixIterator(iterators.CheckpointableIterator):
    """
    Concat items from all given iterators.
    """



    