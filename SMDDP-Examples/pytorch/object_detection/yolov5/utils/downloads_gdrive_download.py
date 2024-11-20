def gdrive_download(id='16TiPfZj7htmTyhntwcZyEEAejOUxuT6m', file='tmp.zip'):
    t = time.time()
    file = Path(file)
    cookie = Path('cookie')
    print(
        f'Downloading https://drive.google.com/uc?export=download&id={id} as {file}... '
        , end='')
    file.unlink(missing_ok=True)
    cookie.unlink(missing_ok=True)
    out = 'NUL' if platform.system() == 'Windows' else '/dev/null'
    os.system(
        f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}'
        )
    if os.path.exists('cookie'):
        s = (
            f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
            )
    else:
        s = (
            f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
            )
    r = os.system(s)
    cookie.unlink(missing_ok=True)
    if r != 0:
        file.unlink(missing_ok=True)
        print('Download error ')
        return r
    if file.suffix == '.zip':
        print('unzipping... ', end='')
        ZipFile(file).extractall(path=file.parent)
        file.unlink()
    print(f'Done ({time.time() - t:.1f}s)')
    return r
