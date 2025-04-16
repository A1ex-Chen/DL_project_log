def check_cls_dataset(dataset, split=''):
    """
    Checks a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'create_self_data', or ''. Defaults to ''.

    Returns:
        (dict): A dictionary containing the following keys:
            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'create_self_data' (Path): The directory path containing the create_self_data set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """
    if str(dataset).startswith(('http:/', 'https:/')):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True,
            delete=False)
    elif Path(dataset).suffix in {'.zip', '.tar', '.gz'}:
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=
            False)
    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else DATASETS_DIR / dataset
        ).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(
            f'\nDataset not found ⚠️, missing path {data_dir}, attempting download...'
            )
        t = time.time()
        if str(dataset) == 'imagenet':
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}",
                shell=True, check=True)
        else:
            url = (
                f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip'
                )
            download(url, dir=data_dir.parent)
        s = f"""Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}
"""
        LOGGER.info(s)
    train_set = data_dir / 'train'
    val_set = data_dir / 'val' if (data_dir / 'val').exists(
        ) else data_dir / 'validation' if (data_dir / 'validation').exists(
        ) else None
    test_set = data_dir / 'create_self_data' if (data_dir / 'create_self_data'
        ).exists() else None
    if split == 'val' and not val_set:
        LOGGER.warning(
            "WARNING ⚠️ Dataset 'split=val' not found, using 'split=create_self_data' instead."
            )
    elif split == 'create_self_data' and not test_set:
        LOGGER.warning(
            "WARNING ⚠️ Dataset 'split=create_self_data' not found, using 'split=val' instead."
            )
    nc = len([x for x in (data_dir / 'train').glob('*') if x.is_dir()])
    names = [x.name for x in (data_dir / 'train').iterdir() if x.is_dir()]
    names = dict(enumerate(sorted(names)))
    for k, v in {'train': train_set, 'val': val_set, 'create_self_data':
        test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob('*.*') if path.suffix[1:].
                lower() in IMG_FORMATS]
            nf = len(files)
            nd = len({file.parent for file in files})
            if nf == 0:
                if k == 'train':
                    raise FileNotFoundError(emojis(
                        f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    LOGGER.warning(
                        f'{prefix} found {nf} images in {nd} classes: WARNING ⚠️ no images found'
                        )
            elif nd != nc:
                LOGGER.warning(
                    f'{prefix} found {nf} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}'
                    )
            else:
                LOGGER.info(f'{prefix} found {nf} images in {nd} classes ✅ ')
    return {'train': train_set, 'val': val_set, 'create_self_data':
        test_set, 'nc': nc, 'names': names}
