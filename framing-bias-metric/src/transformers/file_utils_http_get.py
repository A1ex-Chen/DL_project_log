def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0,
    user_agent: Union[Dict, str, None]=None):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = {'user-agent': http_user_agent(user_agent)}
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    content_length = r.headers.get('Content-Length')
    total = resume_size + int(content_length
        ) if content_length is not None else None
    progress = tqdm(unit='B', unit_scale=True, total=total, initial=
        resume_size, desc='Downloading', disable=bool(logging.get_verbosity
        () == logging.NOTSET))
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()
