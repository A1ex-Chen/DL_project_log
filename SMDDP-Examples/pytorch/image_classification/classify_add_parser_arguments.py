def add_parser_arguments(parser):
    model_names = available_models().keys()
    parser.add_argument('--image-size', default='224', type=int)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
        choices=model_names, help='model architecture: ' + ' | '.join(
        model_names) + ' (default: resnet50)')
    parser.add_argument('--precision', metavar='PREC', default='AMP',
        choices=['AMP', 'FP32'])
    parser.add_argument('--cpu', action='store_true', help=
        'perform inference on CPU')
    parser.add_argument('--image', metavar='<path>', help=
        'path to classified image')
