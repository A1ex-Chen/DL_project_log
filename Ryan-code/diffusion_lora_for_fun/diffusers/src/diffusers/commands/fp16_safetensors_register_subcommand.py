@staticmethod
def register_subcommand(parser: ArgumentParser):
    conversion_parser = parser.add_parser('fp16_safetensors')
    conversion_parser.add_argument('--ckpt_id', type=str, help=
        "Repo id of the checkpoints on which to run the conversion. Example: 'openai/shap-e'."
        )
    conversion_parser.add_argument('--fp16', action='store_true', help=
        'If serializing the variables in FP16 precision.')
    conversion_parser.add_argument('--use_safetensors', action='store_true',
        help='If serializing in the safetensors format.')
    conversion_parser.add_argument('--use_auth_token', action='store_true',
        help=
        'When working with checkpoints having private visibility. When used `huggingface-cli login` needs to be run beforehand.'
        )
    conversion_parser.set_defaults(func=conversion_command_factory)
