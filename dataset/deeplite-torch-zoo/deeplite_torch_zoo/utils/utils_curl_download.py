def curl_download(url, filename, *, silent: bool=False) ->bool:
    """
    Download a file from a url to a filename using curl.
    """
    silent_option = 'sS' if silent else ''
    proc = subprocess.run(['curl', '-#', f'-{silent_option}L', url,
        '--output', filename, '--retry', '9', '-C', '-'])
    return proc.returncode == 0
