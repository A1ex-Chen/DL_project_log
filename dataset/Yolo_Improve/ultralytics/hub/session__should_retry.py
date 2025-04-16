@staticmethod
def _should_retry(status_code):
    """Determines if a request should be retried based on the HTTP status code."""
    retry_codes = {HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.BAD_GATEWAY,
        HTTPStatus.GATEWAY_TIMEOUT}
    return status_code in retry_codes
