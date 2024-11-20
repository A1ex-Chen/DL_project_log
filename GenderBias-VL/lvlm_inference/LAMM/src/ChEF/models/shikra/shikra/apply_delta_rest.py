"""
Usage:
python3 apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/shikra-7b --delta lmsys/shikra-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel
from shikra import ShikraLlamaForCausalLM




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--delta", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base, args.target, args.delta)