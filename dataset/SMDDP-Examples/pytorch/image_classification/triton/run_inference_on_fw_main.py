def main():
    args = _parse_and_validate_args()
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    log_format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    LOGGER.info(f'args:')
    for key, value in vars(args).items():
        LOGGER.info(f'    {key} = {value}')
    Loader: BaseLoader = loaders.get(args.input_type)
    Runner: BaseRunner = runners.get(args.input_type)
    loader = ArgParserGenerator(Loader, module_path=args.input_path).from_args(
        args)
    runner = ArgParserGenerator(Runner).from_args(args)
    LOGGER.info(f'Loading {args.input_path}')
    model = loader.load(args.input_path)
    with runner.init_inference(model=model) as runner_session, NpzWriter(args
        .output_dir) as writer:
        get_dataloader_fn = load_from_file(args.dataloader, label=
            'dataloader', target=DATALOADER_FN_NAME)
        dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
        LOGGER.info(f'Data loader initialized; Running inference')
        for ids, x, y_real in tqdm(dataloader_fn(), unit='batch',
            mininterval=10):
            y_pred = runner_session(x)
            data = _verify_and_format_dump(args, ids=ids, x=x, y_pred=
                y_pred, y_real=y_real)
            writer.write(**data)
        LOGGER.info(f'Inference finished')
