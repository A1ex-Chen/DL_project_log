def parse_option():
    parser = argparse.ArgumentParser('SEEM Interface', add_help=False)
    print(os.getcwd())
    parser.add_argument('--conf_files', default=
        '../VFM/configs/seem/seem_focall_lang.yaml', metavar='FILE', help=
        'path to config file')
    args = parser.parse_args()
    return args
