def _parse_args():
    parser = argparse.ArgumentParser(description=
        'Infer model on Triton server', allow_abbrev=False)
    parser.add_argument('--server-url', type=str, default='localhost:8001',
        help='Inference server URL (default localhost:8001)')
    parser.add_argument('--model-name', help=
        'The name of the model used for inference.', required=True)
    parser.add_argument('--model-version', help=
        'The version of the model used for inference.', required=True)
    parser.add_argument('--dataloader', help=
        'Path to python file containing dataloader.', required=True)
    parser.add_argument('--dump-labels', help='Dump labels to output dir',
        action='store_true', default=False)
    parser.add_argument('--dump-inputs', help='Dump inputs to output dir',
        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help='Verbose logs', action=
        'store_true', default=False)
    parser.add_argument('--output-dir', required=True, help=
        'Path to directory where outputs will be saved')
    parser.add_argument('--response-wait-time', required=False, help=
        'Maximal time to wait for response', default=120)
    parser.add_argument('--max-unresponded-requests', required=False, help=
        'Maximal number of unresponded requests', default=128)
    args, *_ = parser.parse_known_args()
    get_dataloader_fn = load_from_file(args.dataloader, label='dataloader',
        target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)
    args = parser.parse_args()
    return args
