def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18', help=
        'Model name')
    parser.add_argument('--dataset', type=str, default='imagenet', help=
        'Dataset Name')
    parser.add_argument('--num-classes', type=int, default=1000, help=
        'Number of classes')
    parser.add_argument('--pretrained', action='store_true', help=
        'Use pretrained model')
    parser.add_argument('--img-size', type=int, default=224, help=
        'Image size (pixels)')
    parser.add_argument('--num-bytes', type=int, default=1, help=
        'Num Bytes to use in RAM profiling')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')
    return parser.parse_args()
