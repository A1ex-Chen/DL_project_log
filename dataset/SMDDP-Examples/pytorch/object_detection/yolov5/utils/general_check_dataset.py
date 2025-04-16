def check_dataset(data, autodownload=True):
    extract_dir = ''
    if isinstance(data, (str, Path)) and str(data).endswith('.zip'):
        download(data, dir=f'{DATASETS_DIR}/{Path(data).stem}', unzip=True,
            delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob('*.yaml'))
        extract_dir, autodownload = data.parent, False
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)
    for k in ('train', 'val', 'nc'):
        assert k in data, f"data.yaml '{k}:' field missing ❌"
    if 'names' not in data:
        LOGGER.warning(
            "data.yaml 'names:' field missing ⚠️, assigning default names 'class0', 'class1', etc."
            )
        data['names'] = [f'class{i}' for i in range(data['nc'])]
    path = Path(extract_dir or data.get('path') or '')
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    for k in ('train', 'val', 'test'):
        if data.get(k):
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str
                (path / x) for x in data[k]]
    train, val, test, s = (data.get(x) for x in ('train', 'val', 'test',
        'download'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else
            [val])]
        if not all(x.exists() for x in val):
            LOGGER.info('\nDataset not found ⚠️, missing paths %s' % [str(x
                ) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception('Dataset not found ❌')
            t = time.time()
            root = path.parent if 'path' in data else '..'
            if s.startswith('http') and s.endswith('.zip'):
                f = Path(s).name
                LOGGER.info(f'Downloading {s} to {f}...')
                torch.hub.download_url_to_file(s, f)
                Path(root).mkdir(parents=True, exist_ok=True)
                ZipFile(f).extractall(path=root)
                Path(f).unlink()
                r = None
            elif s.startswith('bash '):
                LOGGER.info(f'Running {s} ...')
                r = os.system(s)
            else:
                r = exec(s, {'yaml': data})
            dt = f'({round(time.time() - t, 1)}s)'
            s = f"success ✅ {dt}, saved to {colorstr('bold', root)}" if r in (
                0, None) else f'failure {dt} ❌'
            LOGGER.info(f'Dataset download {s}')
    check_font('Arial.ttf' if is_ascii(data['names']) else
        'Arial.Unicode.ttf', progress=True)
    return data
