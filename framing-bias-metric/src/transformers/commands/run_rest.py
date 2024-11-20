from argparse import ArgumentParser

from transformers.commands import BaseTransformersCLICommand
from transformers.pipelines import SUPPORTED_TASKS, Pipeline, PipelineDataFormat, pipeline

from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name






class RunCommand(BaseTransformersCLICommand):

    @staticmethod
