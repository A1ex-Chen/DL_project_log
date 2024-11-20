def cache_labels(self, path=Path('./labels.cache')):
    """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
    x = {'labels': []}
    nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
    desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
    total = len(self.im_files)
    nkpt, ndim = self.data.get('kpt_shape', (0, 0))
    if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
        raise ValueError(
            "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(func=verify_image_label, iterable=zip(self.
            im_files, self.label_files, repeat(self.prefix), repeat(self.
            use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
            repeat(ndim)))
        pbar = TQDM(results, desc=desc, total=total)
        for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                x['labels'].append({'im_file': im_file, 'shape': shape,
                    'cls': lb[:, 0:1], 'bboxes': lb[:, 1:], 'segments':
                    segments, 'keypoints': keypoint, 'normalized': True,
                    'bbox_format': 'xywh'})
            if msg:
                msgs.append(msg)
            pbar.desc = (
                f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt')
        pbar.close()
    if msgs:
        LOGGER.info('\n'.join(msgs))
    if nf == 0:
        LOGGER.warning(
            f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
    x['hash'] = get_hash(self.label_files + self.im_files)
    x['results'] = nf, nm, ne, nc, len(self.im_files)
    x['msgs'] = msgs
    save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
    return x
