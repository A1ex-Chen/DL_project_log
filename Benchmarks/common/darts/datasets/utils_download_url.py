def download_url(url, root, filename, md5):
    from six.moves import urllib
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)
    makedir_exist_ok(root)
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=
                gen_bar_updater(tqdm(unit='B', unit_scale=True)))
        except Exception:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print(
                    'Failed download. Trying https -> http instead. Downloading '
                     + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath, reporthook=
                    gen_bar_updater(tqdm(unit='B', unit_scale=True)))
