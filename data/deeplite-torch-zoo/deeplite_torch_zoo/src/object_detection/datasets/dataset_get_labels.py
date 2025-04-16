def get_labels(self):
    """Returns dictionary of labels for YOLO training."""
    self.label_files = img2label_paths(self.im_files)
    cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
    try:
        import gc
        gc.disable()
        cache, exists = np.load(str(cache_path), allow_pickle=True).item(
            ), True
        gc.enable()
        assert cache['version'] == self.cache_version
        assert cache['hash'] == get_hash(self.label_files + self.im_files)
    except (FileNotFoundError, AssertionError, AttributeError):
        cache, exists = self.cache_labels(cache_path), False
    nf, nm, ne, nc, n = cache.pop('results')
    if exists and LOCAL_RANK in (-1, 0):
        d = (
            f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            )
        tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=
            TQDM_BAR_FORMAT)
        if cache['msgs']:
            LOGGER.info('\n'.join(cache['msgs']))
    if nf == 0:
        raise FileNotFoundError(
            f'{self.prefix}No labels found in {cache_path}, can not start training.'
            )
    [cache.pop(k) for k in ('hash', 'version', 'msgs')]
    labels = cache['labels']
    self.im_files = [lb['im_file'] for lb in labels]
    lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for
        lb in labels)
    len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
    if len_segments and len_boxes != len_segments:
        LOGGER.warning(
            f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.'
            )
        for lb in labels:
            lb['segments'] = []
    if len_cls == 0:
        raise ValueError(
            f'All labels empty in {cache_path}, can not start training without labels.'
            )
    return labels
