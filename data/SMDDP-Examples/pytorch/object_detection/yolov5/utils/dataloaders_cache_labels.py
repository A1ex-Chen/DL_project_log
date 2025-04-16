def cache_labels(self, path=Path('./labels.cache'), prefix=''):
    x = {}
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
    desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
    with Pool(NUM_THREADS) as pool:
        pbar = tqdm(pool.imap(verify_image_label, zip(self.im_files, self.
            label_files, repeat(prefix))), desc=desc, total=len(self.
            im_files), bar_format=BAR_FORMAT)
        for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x[im_file] = [lb, shape, segments]
            if msg:
                msgs.append(msg)
            pbar.desc = (
                f'{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt')
    pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(
            f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
    x['hash'] = get_hash(self.label_files + self.im_files)
    x['results'] = nf, nm, ne, nc, len(self.im_files)
    x['msgs'] = msgs
    x['version'] = self.cache_version
    try:
        np.save(path, x)
        path.with_suffix('.cache.npy').rename(path)
        LOGGER.info(f'{prefix}New cache created: {path}')
    except Exception as e:
        LOGGER.warning(
            f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}'
            )
    return x
