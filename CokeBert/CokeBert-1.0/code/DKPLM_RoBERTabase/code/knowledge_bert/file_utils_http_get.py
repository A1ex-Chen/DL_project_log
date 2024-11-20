def http_get(url, temp_file, proxies=None, resume_size=0, user_agent=None):
    ua = 'transformers/{}; python/{}'.format(__version__, sys.version.split
        ()[0])
    if isinstance(user_agent, dict):
        ua += '; ' + '; '.join('{}/{}'.format(k, v) for k, v in user_agent.
            items())
    elif isinstance(user_agent, six.string_types):
        ua += '; ' + user_agent
    headers = {'user-agent': ua}
    if resume_size > 0:
        headers['Range'] = 'bytes=%d-' % (resume_size,)
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length
        ) if content_length is not None else None
    progress = tqdm(unit='B', unit_scale=True, total=total, initial=
        resume_size, desc='Downloading', disable=bool(logger.level <=
        logging.INFO))
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()
