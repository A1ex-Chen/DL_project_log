def test_sync_distant_http_model_with_params(run_mocked_service):
    try:
        for lib in _distant_http_model_lib(endpoint_headers={
            'X-Correlation-Id': '123-456-789'}, endpoint_params={'limit': 10}):
            m = lib.get('test_distant_http_model')
            assert m({'some_content': 'something'}) == {'some_content':
                'something', 'limit': 10, 'x_correlation_id': '123-456-789'}
    except Exception:
        _stop_mocked_service_and_print_stderr(run_mocked_service)
        raise
