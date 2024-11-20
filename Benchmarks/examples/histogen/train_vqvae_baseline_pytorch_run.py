def run(params):
    fetch_data(params)
    args = candle.ArgumentStruct(**params)
    dist.launch(config_and_train, args.n_gpu_per_machine, 1, 0, args.
        dist_url, args=(args,))
