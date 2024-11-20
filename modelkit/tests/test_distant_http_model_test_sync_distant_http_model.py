@pytest.mark.parametrize(*test_distant_http_model_args)
def test_sync_distant_http_model(item, headers, params, expected,
    distant_http_model_lib, run_mocked_service):
    try:
        m = distant_http_model_lib.get('test_distant_http_model')
        assert expected == m(item, endpoint_headers=headers,
            endpoint_params=params)
    except Exception:
        _stop_mocked_service_and_print_stderr(run_mocked_service)
        raise
