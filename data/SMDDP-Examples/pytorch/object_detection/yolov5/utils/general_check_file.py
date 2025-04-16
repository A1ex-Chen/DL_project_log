def check_file(file, suffix=''):
    check_suffix(file, suffix)
    file = str(file)
    if Path(file).is_file() or not file:
        return file
    elif file.startswith(('http:/', 'https:/')):
        url = file
        file = Path(urllib.parse.unquote(file).split('?')[0]).name
        if Path(file).is_file():
            LOGGER.info(f'Found {url} locally at {file}')
        else:
            LOGGER.info(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat(
                ).st_size > 0, f'File download failed: {url}'
        return file
    elif file.startswith('clearml://'):
        assert 'clearml' in sys.modules, "ClearML is not installed, so cannot use ClearML dataset. Try running 'pip install clearml'."
        return file
    else:
        files = []
        for d in ('data', 'models', 'utils'):
            files.extend(glob.glob(str(ROOT / d / '**' / file), recursive=True)
                )
        assert len(files), f'File not found: {file}'
        assert len(files
            ) == 1, f"Multiple files match '{file}', specify exact path: {files}"
        return files[0]
