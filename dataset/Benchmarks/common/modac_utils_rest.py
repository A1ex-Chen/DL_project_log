import json
import os

import requests
from tqdm import tqdm
















if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--authenticate",
        action="store_true",
        help="Authenticate MoDaC user and create token",
    )

    args = parser.parse_args()
    if args.authenticate:
        authenticate_modac(generate_token=True)