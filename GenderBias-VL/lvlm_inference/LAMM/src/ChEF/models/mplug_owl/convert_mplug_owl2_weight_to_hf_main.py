def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help=
        'Location of LLaMA_Megatron weights')
    parser.add_argument('--model_size', type=int, default=7, choices=[7, 13,
        30, 65, 70])
    parser.add_argument('--num_input_shards', type=int, default=1)
    parser.add_argument('--num_output_shards', type=int, default=1)
    parser.add_argument('--skip_permute', action='store_true')
    parser.add_argument('--output_dir', help=
        'Location to write HF model and tokenizer')
    args = parser.parse_args()
    write_model(model_path=args.output_dir, input_base_path=args.input_dir,
        model_size=args.model_size, num_input_shards=args.num_input_shards,
        num_output_shards=args.num_output_shards, skip_permute=args.
        skip_permute)
