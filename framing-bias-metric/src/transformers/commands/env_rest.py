import platform
from argparse import ArgumentParser

from transformers import __version__ as version
from transformers import is_tf_available, is_torch_available
from transformers.commands import BaseTransformersCLICommand




class EnvironmentCommand(BaseTransformersCLICommand):
    @staticmethod


    @staticmethod