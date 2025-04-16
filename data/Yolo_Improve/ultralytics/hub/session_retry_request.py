def retry_request():
    """Attempts to call `request_func` with retries, timeout, and optional threading."""
    t0 = time.time()
    response = None
    for i in range(retry + 1):
        if time.time() - t0 > timeout:
            LOGGER.warning(f'{PREFIX}Timeout for request reached. {HELP_MSG}')
            break
        response = request_func(*args, **kwargs)
        if response is None:
            LOGGER.warning(
                f'{PREFIX}Received no response from the request. {HELP_MSG}')
            time.sleep(2 ** i)
            continue
        if progress_total:
            self._show_upload_progress(progress_total, response)
        elif stream_response:
            self._iterate_content(response)
        if HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
            if kwargs.get('metrics'):
                self.metrics_upload_failed_queue = {}
            return response
        if i == 0:
            message = self._get_failure_message(response, retry, timeout)
            if verbose:
                LOGGER.warning(
                    f'{PREFIX}{message} {HELP_MSG} ({response.status_code})')
        if not self._should_retry(response.status_code):
            LOGGER.warning(
                f'{PREFIX}Request failed. {HELP_MSG} ({response.status_code}')
            break
        time.sleep(2 ** i)
    if response is None and kwargs.get('metrics'):
        self.metrics_upload_failed_queue.update(kwargs.get('metrics', None))
    return response
