def eval_parser():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg-path', help='path to configuration file.')
    parser.add_argument('--name', type=str, default='A2', help=
        'evaluation name')
    parser.add_argument('--ckpt', type=str, help='path to configuration file.')
    parser.add_argument('--eval_opt', type=str, default='all', help=
        'path to configuration file.')
    parser.add_argument('--lora_r', type=int, default=64, help=
        'lora rank of the model')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha'
        )
    parser.add_argument('--options', nargs='+', help=
        'override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.'
        )
    return parser
