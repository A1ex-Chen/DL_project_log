def download_file(session, file_spec, stats, chunk_size=128, num_attempts=
    10, **kwargs):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size << 10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats['lock']:
                            stats['bytes_done'] += len(chunk)
            if 'file_size' in file_spec and data_size != file_spec['file_size'
                ]:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec[
                'file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            if 'pixel_size' in file_spec or 'pixel_md5' in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if 'pixel_size' in file_spec and list(image.size
                        ) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_path)
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)
                        ).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_path)
            break
        except:
            with stats['lock']:
                stats['bytes_done'] -= data_size
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                data_str = data.decode('utf-8')
                links = [html.unescape(link) for link in data_str.split('"'
                    ) if 'export=download' in link]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue
                if 'Google Drive - Quota exceeded' in data_str:
                    if not attempts_left:
                        raise IOError(
                            'Google Drive download quota exceeded -- please try again later'
                            )
            if not attempts_left:
                raise
    os.replace(tmp_path, file_path)
    with stats['lock']:
        stats['files_done'] += 1
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass
