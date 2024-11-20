def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """
    file = check_file(dataset)
    extract_dir = ''
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=
            False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False
    data = yaml_load(file, append_filename=True)
    for k in ('train', 'val'):
        if k not in data:
            if k != 'val' or 'validation' not in data:
                raise SyntaxError(emojis(
                    f"""{dataset} '{k}:' key missing ❌.
'train' and 'val' are required in all data YAMLs."""
                    ))
            LOGGER.info(
                "WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format."
                )
            data['val'] = data.pop('validation')
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(
            f"""{dataset} key missing ❌.
 either 'names' or 'nc' are required in all data YAMLs."""
            ))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(
            f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."
            ))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])
    data['names'] = check_class_names(data['names'])
    path = Path(extract_dir or data.get('path') or Path(data.get(
        'yaml_file', '')).parent)
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()
    data['path'] = path
    for k in ('train', 'val', 'create_self_data', 'minival'):
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    val, s = (data.get(x) for x in ('val', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else
            [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            m = f"""
Dataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"""
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"""
Note dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"""
                raise FileNotFoundError(m)
            t = time.time()
            r = None
            if s.startswith('http') and s.endswith('.zip'):
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith('bash '):
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:
                exec(s, {'yaml': data})
            dt = f'({round(time.time() - t, 1)}s)'
            s = (
                f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if
                r in {0, None} else f'failure {dt} ❌')
            LOGGER.info(f'Dataset download {s}\n')
    check_font('Arial.ttf' if is_ascii(data['names']) else 'Arial.Unicode.ttf')
    return data
