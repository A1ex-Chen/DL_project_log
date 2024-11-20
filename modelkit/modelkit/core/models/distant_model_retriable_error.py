def retriable_error(exception):
    return isinstance(exception, aiohttp.client_exceptions.ClientConnectorError
        ) or isinstance(exception, requests.exceptions.ConnectionError)
