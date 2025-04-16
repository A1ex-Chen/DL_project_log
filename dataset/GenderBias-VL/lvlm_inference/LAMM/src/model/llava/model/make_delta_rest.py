"""
Usage:
python3 -m llava.model.make_delta --base ~/model_weights/llama-7b --target ~/model_weights/llava-7b --delta ~/model_weights/llava-7b-delta --hub-repo-id liuhaotian/llava-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava.model.utils import auto_upgrade




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path, args.hub_repo_id)