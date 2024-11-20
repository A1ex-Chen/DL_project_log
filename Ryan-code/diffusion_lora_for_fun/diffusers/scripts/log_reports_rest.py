import argparse
import json
import os
from datetime import date
from pathlib import Path

from slack_sdk import WebClient
from tabulate import tabulate


MAX_LEN_MESSAGE = 2900  # slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="diffusers-ci-nightly")




if __name__ == "__main__":
    args = parser.parse_args()
    main(args.slack_channel_name)