def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument('--cfg', dest='config_file', default='', metavar=
        'FILE', help='path to config file', type=str)
    parser.add_argument('opts', help=
        'Modify config options using the command-line', default=None, nargs
        =argparse.REMAINDER)
    parser.add_argument('--vfmlab', action='store_true', help=
        'Whether to save the VFM generated labels.')
    args = parser.parse_args()
    args.vfmlab = True
    return args
