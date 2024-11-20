def parse_option():
    parser = argparse.ArgumentParser('SEEM Demo', add_help=False)
    parser.add_argument('--conf_files', default=
        'configs/seem/seem_focall_lang.yaml', metavar='FILE', help=
        'path to config file')
    args = parser.parse_args()
    return args
