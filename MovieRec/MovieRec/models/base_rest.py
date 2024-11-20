import torch.nn as nn

from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):

    @classmethod
    @abstractmethod
