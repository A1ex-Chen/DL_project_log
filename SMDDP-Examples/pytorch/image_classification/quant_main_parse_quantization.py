def parse_quantization(parser):
    model_names = available_models().keys()
    parser.add_argument('--arch', '-a', metavar='ARCH', default=
        'efficientnet-quant-b0', choices=model_names, help=
        'model architecture: ' + ' | '.join(model_names) +
        ' (default: efficientnet-quant-b0)')
    parser.add_argument('--skip-calibration', action='store_true', help=
        'skip calibration before training, (default: false)')
