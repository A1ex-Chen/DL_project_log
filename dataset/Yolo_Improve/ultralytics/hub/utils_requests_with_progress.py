def requests_with_progress(method, url, **kwargs):
    """
    Make an HTTP request using the specified method and URL, with an optional progress bar.

    Args:
        method (str): The HTTP method to use (e.g. 'GET', 'POST').
        url (str): The URL to send the request to.
        **kwargs (any): Additional keyword arguments to pass to the underlying `requests.request` function.

    Returns:
        (requests.Response): The response object from the HTTP request.

    Note:
        - If 'progress' is set to True, the progress bar will display the download progress for responses with a known
        content length.
        - If 'progress' is a number then progress bar will display assuming content length = progress.
    """
    progress = kwargs.pop('progress', False)
    if not progress:
        return requests.request(method, url, **kwargs)
    response = requests.request(method, url, stream=True, **kwargs)
    total = int(response.headers.get('content-length', 0) if isinstance(
        progress, bool) else progress)
    try:
        pbar = TQDM(total=total, unit='B', unit_scale=True, unit_divisor=1024)
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
        pbar.close()
    except requests.exceptions.ChunkedEncodingError:
        response.close()
    return response
