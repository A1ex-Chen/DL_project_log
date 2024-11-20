def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-b', default=64, type=int, help=
        'mini-batch size per process (default: 256)')
    parser.add_argument('-o', help='output files path', default='samples/')
    parser.add_argument('--checkpoint', required=True, type=str, help=
        'saved model to sample from')
    parser.add_argument('-n', type=int, default=64, help=
        'number of samples to draw')
    parser.add_argument('--image', action='store_true', help=
        'save images instead of numpy array')
    return parser.parse_args()
