def parse_config():
    config: Config = Config()
    parser = argparse.ArgumentParser(description=
        'Simple example of a training script.')
    parser.add_argument('--project', '-pj', type=str, default=None,
        required=True, help='Name of the project')
    parser.add_argument('--ds_root', '-dsr', type=str, default=None,
        required=True, help='Root of dataset')
    parser.add_argument('--repeat', '-r', type=int, default=None, required=
        True, help='The repetition of data points')
    parser.add_argument('--model_type', '-mt', type=str, default=None,
        required=True, help='The detection model type')
    parser.add_argument('--model_id', '-mi', type=str, default=None,
        required=True, help='The detection model ID')
    parser.add_argument('--loss_metric', '-lm', type=str, default=None,
        required=True, help=
        'The distance metric between ground-truth and predicted image')
    parser.add_argument('--lr', '-lr', type=float, default=None, required=
        True, help='The learning rate of optimizer')
    parser.add_argument('--max_iter', '-it', type=int, default=None,
        required=True, help='The maximum iteration number')
    parser.add_argument('--num_cycles', '-nc', type=int, default=None,
        required=True, help='The nnumber of learning rate scheduler cycle')
    parser.add_argument('--device', '-d', type=str, default=None, required=
        True, help='The computing unit that is used')
    args = parser.parse_args()
    config.project = args.project
    config.ds_root = args.ds_root
    config.repeat = args.repeat
    config.model_type = args.model_type
    config.model_id = args.model_id
    config.loss_metric = args.loss_metric
    config.lr = args.lr
    config.max_iter = args.max_iter
    config.num_cycles = args.num_cycles
    config.device = args.device
    config.re_init__()
    return config
