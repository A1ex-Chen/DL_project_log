def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-w', '--workers', default=16, type=int, metavar=
        'N', help='mini-batch size per process (default: 256)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('-g', '--grad-clip', default=2.0, type=float,
        metavar='N', help='mini-batch size per process (default: 256)')
    parser.add_argument('-d', default='moses/data', help=
        'folder with train and test smiles files')
    parser.add_argument('-mp', help='model save path', default='models/')
    parser.add_argument('-op', help='output files path', default='output/')
    parser.add_argument('--checkpoint', default=None, type=str)
    return parser.parse_args()
