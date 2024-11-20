def download_file(url, dest_folder, fname, overwrite=False):
    fpath = os.path.join(dest_folder, fname)
    if os.path.isfile(fpath):
        if overwrite:
            print('Overwriting existing file')
        else:
            print('File exists, skipping download.')
            return
    tmp_fpath = fpath + '.tmp'
    r = requests.get(url, stream=True)
    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024 * 1024
    total_chunks = int(file_size / chunk_size)
    with open(tmp_fpath, 'wb') as fp:
        content_iterator = r.iter_content(chunk_size=chunk_size)
        chunks = tqdm.tqdm(content_iterator, total=total_chunks, unit='MB',
            desc=fpath, leave=True)
        for chunk in chunks:
            fp.write(chunk)
    os.rename(tmp_fpath, fpath)
