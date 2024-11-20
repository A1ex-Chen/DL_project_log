from argparse import ArgumentParser, Namespace

from transformers.commands import BaseTransformersCLICommand

from ..utils import logging




IMPORT_ERROR_MESSAGE = """
transformers can only be used from the commandline to convert TensorFlow models in PyTorch, In that case, it requires
TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.
"""


class ConvertCommand(BaseTransformersCLICommand):
    @staticmethod

