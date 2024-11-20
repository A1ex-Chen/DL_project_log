def verify_images(self):
    """Verify all images in dataset."""
    desc = f'{self.prefix}Scanning {self.root}...'
    path = Path(self.root).with_suffix('.cache')
    with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError
        ):
        cache = load_dataset_cache_file(path)
        assert cache['version'] == DATASET_CACHE_VERSION
        assert cache['hash'] == get_hash([x[0] for x in self.samples])
        nf, nc, n, samples = cache.pop('results')
        if LOCAL_RANK in {-1, 0}:
            d = f'{desc} {nf} images, {nc} corrupt'
            TQDM(None, desc=d, total=n, initial=n)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))
        return samples
    nf, nc, msgs, samples, x = 0, 0, [], [], {}
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(func=verify_image, iterable=zip(self.samples,
            repeat(self.prefix)))
        pbar = TQDM(results, desc=desc, total=len(self.samples))
        for sample, nf_f, nc_f, msg in pbar:
            if nf_f:
                samples.append(sample)
            if msg:
                msgs.append(msg)
            nf += nf_f
            nc += nc_f
            pbar.desc = f'{desc} {nf} images, {nc} corrupt'
        pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    x['hash'] = get_hash([x[0] for x in self.samples])
    x['results'] = nf, nc, len(samples), samples
    x['msgs'] = msgs
    save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
    return samples
