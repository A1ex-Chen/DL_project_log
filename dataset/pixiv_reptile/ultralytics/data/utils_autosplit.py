def autosplit(path=DATASETS_DIR / 'coco8/images', weights=(0.9, 0.1, 0.0),
    annotated_only=False):
    """
    Automatically split a dataset into train/val/create_self_data splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.
        weights (list | tuple, optional): Train, validation, and create_self_data split fractions. Defaults to (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.

    Example:
        ```python
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """
    path = Path(path)
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in
        IMG_FORMATS)
    n = len(files)
    random.seed(0)
    indices = random.choices([0, 1, 2], weights=weights, k=n)
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()
    LOGGER.info(f'Autosplitting images from {path}' + 
        ', using *.txt labeled images only' * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')
