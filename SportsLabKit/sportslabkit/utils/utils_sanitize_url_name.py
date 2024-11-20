def sanitize_url_name(url: str) ->str:
    """Sanitize the URL to create a safe filename for caching.

    Args:
        url (str): The URL to sanitize.

    Returns:
        str: A sanitized version of the URL suitable for use as a filename.
    """
    parsed_url = urlparse(url)
    filename = Path(parsed_url.path).name
    sanitized_name = re.sub('[^\\w\\s-]', '', filename).strip().replace(' ',
        '_')
    return sanitized_name
