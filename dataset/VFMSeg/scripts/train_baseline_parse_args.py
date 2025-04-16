def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument('--cfg', dest='config_file', default='', metavar=
        'FILE', help='path to config file', type=str)
    parser.add_argument('opts', help=
        'Modify config options using the command-line', default=None, nargs
        =argparse.REMAINDER)
    args = parser.parse_args()
    return args
