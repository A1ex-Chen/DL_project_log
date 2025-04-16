def _get_args():
    parser = argparse.ArgumentParser(description=
        'Script for conversion between model formats.', allow_abbrev=False)
    parser.add_argument('--input-path', help=
        'Path to input model file (python module or binary file)', required
        =True)
    parser.add_argument('--input-type', help='Input model type', choices=[f
        .value for f in INPUT_MODEL_TYPES], required=True)
    parser.add_argument('--output-path', help='Path to output model file',
        required=True)
    parser.add_argument('--output-type', help='Output model type', choices=
        [f.value for f in OUTPUT_MODEL_TYPES], required=True)
    parser.add_argument('--dataloader', help=
        'Path to python module containing data loader')
    parser.add_argument('-v', '--verbose', help='Verbose logs', action=
        'store_true', default=False)
    parser.add_argument('--ignore-unknown-parameters', help=
        'Ignore unknown parameters (argument often used in CI where set of arguments is constant)'
        , action='store_true', default=False)
    args, unparsed_args = parser.parse_known_args()
    Loader: BaseLoader = loaders.get(args.input_type)
    ArgParserGenerator(Loader, module_path=args.input_path).update_argparser(
        parser)
    converter_name = f'{args.input_type}--{args.output_type}'
    Converter: BaseConverter = converters.get(converter_name)
    if Converter is not None:
        ArgParserGenerator(Converter).update_argparser(parser)
    Saver: BaseSaver = savers.get(args.output_type)
    ArgParserGenerator(Saver).update_argparser(parser)
    if args.dataloader is not None:
        get_dataloader_fn = load_from_file(args.dataloader, label=
            'dataloader', target=DATALOADER_FN_NAME)
        ArgParserGenerator(get_dataloader_fn).update_argparser(parser)
    if args.ignore_unknown_parameters:
        args, unknown_args = parser.parse_known_args()
        LOGGER.warning(f'Got additional args {unknown_args}')
    else:
        args = parser.parse_args()
    return args
