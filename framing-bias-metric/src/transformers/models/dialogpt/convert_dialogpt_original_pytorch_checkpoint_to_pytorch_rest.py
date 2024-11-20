import argparse
import os

import torch

from transformers.file_utils import WEIGHTS_NAME


DIALOGPT_MODELS = ["small", "medium", "large"]

OLD_KEY = "lm_head.decoder.weight"
NEW_KEY = "lm_head.weight"




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogpt_path", default=".", type=str)
    args = parser.parse_args()
    for MODEL in DIALOGPT_MODELS:
        checkpoint_path = os.path.join(args.dialogpt_path, f"{MODEL}_ft.pkl")
        pytorch_dump_folder_path = f"./DialoGPT-{MODEL}"
        convert_dialogpt_checkpoint(
            checkpoint_path,
            pytorch_dump_folder_path,
        )