#!/usr/bin/env python
from argparse import ArgumentParser

from transformers.commands.add_new_model import AddNewModelCommand
from transformers.commands.convert import ConvertCommand
from transformers.commands.download import DownloadCommand
from transformers.commands.env import EnvironmentCommand
from transformers.commands.run import RunCommand
from transformers.commands.serving import ServeCommand
from transformers.commands.user import UserCommands




if __name__ == "__main__":
    main()