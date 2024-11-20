def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to the sentencepiece model')
    parser.add_argument('JSONS', nargs='+', help=
        'Json files with .[].transcript field')
    parser.add_argument('--output_dir', default=None, help=
        'If set, output files will be in a different directory')
    parser.add_argument('--suffix', default='-tokenized', help=
        'Suffix added to the output files')
    parser.add_argument('--output_format', default='pkl', choices=['pkl',
        'json'], help='Output format')
    return parser
