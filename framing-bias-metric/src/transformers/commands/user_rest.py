import os
import subprocess
import sys
from argparse import ArgumentParser
from getpass import getpass
from typing import List, Union

from requests.exceptions import HTTPError
from transformers.commands import BaseTransformersCLICommand
from transformers.hf_api import HfApi, HfFolder


UPLOAD_MAX_FILES = 15


class UserCommands(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser("login", help="Log in using the same credentials as on huggingface.co")
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser("whoami", help="Find out which huggingface.co account you are logged in as.")
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser("logout", help="Log out")
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))
        # s3_datasets (s3-based system)
        s3_parser = parser.add_parser(
            "s3_datasets", help="{ls, rm} Commands to interact with the files you upload on S3."
        )
        s3_subparsers = s3_parser.add_subparsers(help="s3 related commands")
        ls_parser = s3_subparsers.add_parser("ls")
        ls_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        ls_parser.set_defaults(func=lambda args: ListObjsCommand(args))
        rm_parser = s3_subparsers.add_parser("rm")
        rm_parser.add_argument("filename", type=str, help="individual object filename to delete from huggingface.co.")
        rm_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        rm_parser.set_defaults(func=lambda args: DeleteObjCommand(args))
        upload_parser = s3_subparsers.add_parser("upload", help="Upload a file to S3.")
        upload_parser.add_argument("path", type=str, help="Local path of the folder or individual file to upload.")
        upload_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        upload_parser.add_argument(
            "--filename", type=str, default=None, help="Optional: override individual object filename on S3."
        )
        upload_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        upload_parser.set_defaults(func=lambda args: UploadCommand(args))
        # deprecated model upload
        upload_parser = parser.add_parser(
            "upload",
            help=(
                "Deprecated: used to be the way to upload a model to S3."
                " We now use a git-based system for storing models and other artifacts."
                " Use the `repo create` command instead."
            ),
        )
        upload_parser.set_defaults(func=lambda args: DeprecatedUploadCommand(args))

        # new system: git-based repo system
        repo_parser = parser.add_parser(
            "repo", help="{create, ls-files} Commands to interact with your huggingface.co repos."
        )
        repo_subparsers = repo_parser.add_subparsers(help="huggingface.co repos related commands")
        ls_parser = repo_subparsers.add_parser("ls-files", help="List all your files on huggingface.co")
        ls_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        ls_parser.set_defaults(func=lambda args: ListReposObjsCommand(args))
        repo_create_parser = repo_subparsers.add_parser("create", help="Create a new repo on huggingface.co")
        repo_create_parser.add_argument(
            "name",
            type=str,
            help="Name for your model's repo. Will be namespaced under your username to build the model id.",
        )
        repo_create_parser.add_argument("--organization", type=str, help="Optional: organization namespace.")
        repo_create_parser.add_argument("-y", "--yes", action="store_true", help="Optional: answer Yes to the prompt")
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _gray = "\u001b[90m"
    _reset = "\u001b[0m"

    @classmethod
    def bold(cls, s):
        return "{}{}{}".format(cls._bold, s, cls._reset)

    @classmethod
    def red(cls, s):
        return "{}{}{}".format(cls._bold + cls._red, s, cls._reset)

    @classmethod
    def gray(cls, s):
        return "{}{}{}".format(cls._gray, s, cls._reset)




class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """

    _bold = "\u001b[1m"
    _red = "\u001b[31m"
    _gray = "\u001b[90m"
    _reset = "\u001b[0m"

    @classmethod

    @classmethod

    @classmethod


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    col_widths = [max(len(str(x)) for x in col) for col in zip(*rows, headers)]
    row_format = ("{{:{}}} " * len(headers)).format(*col_widths)
    lines = []
    lines.append(row_format.format(*headers))
    lines.append(row_format.format(*["-" * w for w in col_widths]))
    for row in rows:
        lines.append(row_format.format(*row))
    return "\n".join(lines)


class BaseUserCommand:


class LoginCommand(BaseUserCommand):


class WhoamiCommand(BaseUserCommand):


class LogoutCommand(BaseUserCommand):


class ListObjsCommand(BaseUserCommand):


class DeleteObjCommand(BaseUserCommand):


class ListReposObjsCommand(BaseUserCommand):


class RepoCreateCommand(BaseUserCommand):


class DeprecatedUploadCommand(BaseUserCommand):


class UploadCommand(BaseUserCommand):
