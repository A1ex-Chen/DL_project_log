def parallel_preprocess(dataset, input_dir, dest_dir, target_sr, speed,
    overwrite, parallel):
    with multiprocessing.Pool(parallel) as p:
        func = functools.partial(preprocess, input_dir=input_dir, dest_dir=
            dest_dir, target_sr=target_sr, speed=speed, overwrite=overwrite)
        dataset = list(tqdm(p.imap(func, dataset), total=len(dataset)))
        return dataset
