def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match('^(?:http)s?://', input_url, re.IGNORECASE) is not None
    return is_url
