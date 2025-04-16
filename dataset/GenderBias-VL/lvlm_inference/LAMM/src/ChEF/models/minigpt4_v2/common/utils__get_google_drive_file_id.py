def _get_google_drive_file_id(url: str) ->Optional[str]:
    parts = urlparse(url)
    if re.match('(drive|docs)[.]google[.]com', parts.netloc) is None:
        return None
    match = re.match('/file/d/(?P<id>[^/]*)', parts.path)
    if match is None:
        return None
    return match.group('id')
