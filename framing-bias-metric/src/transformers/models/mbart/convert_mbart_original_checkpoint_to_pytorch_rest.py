import argparse

import torch

from transformers import BartForConditionalGeneration, MBartConfig

from ..bart.convert_bart_original_pytorch_checkpoint_to_pytorch import remove_ignore_keys_




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config",
        default="facebook/mbart-large-cc25",
        type=str,
        help="Which huggingface architecture to use: bart-large-xsum",
    )
    args = parser.parse_args()
    model = convert_fairseq_mbart_checkpoint_from_disk(args.fairseq_path, hf_config_path=args.hf_config)
    model.save_pretrained(args.pytorch_dump_folder_path)