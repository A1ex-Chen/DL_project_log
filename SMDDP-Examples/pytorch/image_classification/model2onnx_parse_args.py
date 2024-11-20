def parse_args(parser):
    """
    Parse commandline arguments.
    """
    model_names = available_models().keys()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
        choices=model_names, help='model architecture: ' + ' | '.join(
        model_names) + ' (default: resnet50)')
    parser.add_argument('--device', metavar='DEVICE', default='cuda',
        choices=['cpu', 'cuda'], help=
        'device on which model is settled: cpu, cuda (default: cuda)')
    parser.add_argument('--image-size', default=None, type=int, help=
        'resolution of image')
    parser.add_argument('--output', type=str, help='Path to converted model')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256) per gpu')
    return parser
