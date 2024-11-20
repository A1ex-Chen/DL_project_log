def test_json_logging(monkeypatch):
    monkeypatch.setenv('DEV_LOGGING', '')
    with capture_logs() as cap_logs:
        logger = structlog.get_logger('testing')
        logger.info('It works', even_with='structured fields')
    d = cap_logs[0]
    assert d['even_with'] == 'structured fields'
    assert d['event'] == 'It works'
