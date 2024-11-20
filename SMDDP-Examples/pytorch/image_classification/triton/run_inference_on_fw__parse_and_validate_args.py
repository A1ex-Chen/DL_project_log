def _parse_and_validate_args():
    supported_inputs = set(runners.supported_extensions) & set(loaders.
        supported_extensions)
    parser = argparse.ArgumentParser(description=
        'Dump local inference output of given model', allow_abbrev=False)
    parser.add_argument('--input-path', help='Path to input model',
        required=True)
    parser.add_argument('--input-type', help='Input model type', choices=
        supported_inputs, required=True)
    parser.add_argument('--dataloader', help=
        'Path to python file containing dataloader.', required=True)
    parser.add_argument('--output-dir', help=
        'Path to dir where output files will be stored', required=True)
    parser.add_argument('--dump-labels', help='Dump labels to output dir',
        action='store_true', default=False)
    parser.add_argument('--dump-inputs', help='Dump inputs to output dir',
        action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help='Verbose logs', action=
        'store_true', default=False)
    args, *_ = parser.parse_known_args()
    get_dataloader_fn = load_from_file(args.dataloader, label='dataloader',
        target=DATALOADER_FN_NAME)
    ArgParserGenerator(get_dataloader_fn).update_argparser(parser)
    Loader: BaseLoader = loaders.get(args.input_type)
    ArgParserGenerator(Loader, module_path=args.input_path).update_argparser(
        parser)
    Runner: BaseRunner = runners.get(args.input_type)
    ArgParserGenerator(Runner).update_argparser(parser)
    args = parser.parse_args()
    types_requiring_io_params = []
    if args.input_type in types_requiring_io_params and not all(p for p in
        [args.inputs, args.outputs]):
        parser.error(
            f'For {args.input_type} input provide --inputs and --outputs parameters'
            )
    return args
