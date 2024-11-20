import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from cookiecutter.main import cookiecutter
from transformers.commands import BaseTransformersCLICommand

from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class AddNewModelCommand(BaseTransformersCLICommand):
    @staticmethod



        if output_pytorch:
            if not self._testing:
                remove_copy_lines(f"{directory}/modeling_{lowercase_model_name}.py")

            shutil.move(
                f"{directory}/modeling_{lowercase_model_name}.py",
                f"{model_dir}/modeling_{lowercase_model_name}.py",
            )

            shutil.move(
                f"{directory}/test_modeling_{lowercase_model_name}.py",
                f"{path_to_transformer_root}/tests/test_modeling_{lowercase_model_name}.py",
            )
        else:
            os.remove(f"{directory}/modeling_{lowercase_model_name}.py")
            os.remove(f"{directory}/test_modeling_{lowercase_model_name}.py")

        if output_tensorflow:
            if not self._testing:
                remove_copy_lines(f"{directory}/modeling_tf_{lowercase_model_name}.py")

            shutil.move(
                f"{directory}/modeling_tf_{lowercase_model_name}.py",
                f"{model_dir}/modeling_tf_{lowercase_model_name}.py",
            )

            shutil.move(
                f"{directory}/test_modeling_tf_{lowercase_model_name}.py",
                f"{path_to_transformer_root}/tests/test_modeling_tf_{lowercase_model_name}.py",
            )
        else:
            os.remove(f"{directory}/modeling_tf_{lowercase_model_name}.py")
            os.remove(f"{directory}/test_modeling_tf_{lowercase_model_name}.py")

        shutil.move(
            f"{directory}/{lowercase_model_name}.rst",
            f"{path_to_transformer_root}/docs/source/model_doc/{lowercase_model_name}.rst",
        )

        shutil.move(
            f"{directory}/tokenization_{lowercase_model_name}.py",
            f"{model_dir}/tokenization_{lowercase_model_name}.py",
        )

        from os import fdopen, remove
        from shutil import copymode, move
        from tempfile import mkstemp




        replace_in_files(f"{directory}/to_replace_{lowercase_model_name}.py")
        os.rmdir(directory)