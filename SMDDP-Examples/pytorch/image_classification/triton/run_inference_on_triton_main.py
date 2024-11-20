def main():
    args = _parse_args()
    log_format = '%(asctime)s %(levelname)s %(name)s %(message)s'
    log_level = logging.INFO if not args.verbose else logging.DEBUG
    logging.basicConfig(level=log_level, format=log_format)
    LOGGER.info(f'args:')
    for key, value in vars(args).items():
        LOGGER.info(f'    {key} = {value}')
    get_dataloader_fn = load_from_file(args.dataloader, label='dataloader',
        target=DATALOADER_FN_NAME)
    dataloader_fn = ArgParserGenerator(get_dataloader_fn).from_args(args)
    runner = AsyncGRPCTritonRunner(args.server_url, args.model_name, args.
        model_version, dataloader=dataloader_fn(), verbose=False,
        resp_wait_s=args.response_wait_time, max_unresponded_reqs=args.
        max_unresponded_requests)
    with NpzWriter(output_dir=args.output_dir) as writer:
        start = time.time()
        for ids, x, y_pred, y_real in tqdm(runner, unit='batch', mininterval=10
            ):
            data = _verify_and_format_dump(args, ids, x, y_pred, y_real)
            writer.write(**data)
        stop = time.time()
    LOGGER.info(f'\nThe inference took {stop - start:0.3f}s')
