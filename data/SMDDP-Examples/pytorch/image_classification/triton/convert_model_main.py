def main():
    args = _get_args()
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    LOGGER.info(f'args:')
    for key, value in vars(args).items():
        LOGGER.info(f'    {key} = {value}')
    requested_model_precision = Precision(args.precision)
    dataloader_fn = None
    converter_name = f'{args.input_type}--{args.output_type}'
    Converter: BaseConverter = converters.get(converter_name)
    if Converter:
        args.precision = Converter.required_source_model_precision(
            requested_model_precision).value
    Loader: BaseLoader = loaders.get(args.input_type)
    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(
        args)
    model = loader.load(args.input_path)
    LOGGER.info('inputs: %s', model.inputs)
    LOGGER.info('outputs: %s', model.outputs)
    if Converter:
        if args.dataloader is not None:
            get_dataloader_fn = load_from_file(args.dataloader, label=
                'dataloader', target=DATALOADER_FN_NAME)
            dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(
                args)
    args.precision = requested_model_precision.value
    if Converter:
        converter = ArgParserGenerator(Converter).from_args(args)
        model = converter.convert(model, dataloader_fn=dataloader_fn)
    Saver: BaseSaver = savers.get(args.output_type)
    saver = ArgParserGenerator(Saver).from_args(args)
    saver.save(model, args.output_path)
    return 0
