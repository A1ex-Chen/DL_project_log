import numpy as np
import torch

from bitnet.at import AutoregressiveWrapper
from bitnet.bit_transformer import BitNetTransformer


class BitNetInference:
    """
    A class used to perform inference with the BitNetTransformer model.

    ...

    Attributes
    ----------
    model : torch.nn.Module
        an instance of the BitNetTransformer model
    device : str
        the device to run the model on ('cpu' or 'cuda')

    Methods
    -------
    load_model(model_path)
        Loads a trained model from a .pth file.
    generate(input_str, length)
        Generates a sequence of tokens based on the input string.
    """



    @staticmethod

    @staticmethod
