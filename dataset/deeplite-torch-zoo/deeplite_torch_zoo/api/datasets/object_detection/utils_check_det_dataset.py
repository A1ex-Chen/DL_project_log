def check_det_dataset(dataset, data_root=None, autodownload=True):
    """Download, check and/or unzip dataset if not found locally."""
    data = check_file(dataset)
    extract_dir = ''
    if isinstance(data, (str, Path)) and (zipfile.is_zipfile(data) or
        is_tarfile(data)):
        new_dir = safe_download(data, dir=DATASETS_DIR, unzip=True, delete=
            False, curl=False)
        data = next((DATASETS_DIR / new_dir).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False
    if isinstance(data, (str, Path)):
        data = yaml_load(data, append_filename=True)
    if data_root is not None:
        if not isinstance(data_root, (str, Path)):
            raise ValueError(f'Invalid data path provided: {data_root}')
        data['path'] = data_root
    for k in ('train', 'val'):
        if k not in data:
            raise SyntaxError(emojis(
                f"""{dataset} '{k}:' key missing ❌.
'train' and 'val' are required in all data YAMLs."""
                ))
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
    if not path.is_absolute() and data_root is None:
        path = (DATASETS_DIR / path).resolve()
    download_dir = path
    path = path / dataset.stem
    data['path'] = path
    for k in ('train', 'val', 'test'):
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]
    _, val, _, s = (data.get(x) for x in ('train', 'val', 'test', 'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else
            [val])]
        if not all(x.exists() for x in val):
            name = clean_url(dataset)
            m = f"\nDataset '{name}' images not found ⚠️, missing paths %s" % [
                str(x) for x in val if not x.exists()]
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{download_dir}'"
                raise FileNotFoundError(m)
            t = time.time()
            if s.startswith('http') and s.endswith('.zip'):
                safe_download(url=s, dir=download_dir, delete=True)
                r = None
            elif s.startswith('bash '):
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:
                r = exec(s, {'yaml': data})
            dt = f'({round(time.time() - t, 1)}s)'
            s = (
                f"success ✅ {dt}, saved to {colorstr('bold', download_dir)}" if
                r in (0, None) else f'failure {dt} ❌')
            LOGGER.info(f'Dataset download {s}\n')
    return data
