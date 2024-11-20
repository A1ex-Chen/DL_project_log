def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--pretrained_model_name_or_path', type=str,
        default=None, required=True, help=
        'Path to pretrained model or model identifier from huggingface.co/models.'
        )
    parser.add_argument('-c', '--caption', type=str, default=
        'robotic cat with wings', help='Text used to generate images.')
    parser.add_argument('-n', '--images_num', type=int, default=4, help=
        'How much images to generate.')
    parser.add_argument('-s', '--seed', type=int, default=42, help=
        'Seed for random process.')
    parser.add_argument('-ci', '--cuda_id', type=int, default=0, help=
        'cuda_id.')
    args = parser.parse_args()
    return args
