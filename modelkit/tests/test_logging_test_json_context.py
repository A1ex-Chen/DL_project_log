def test_json_context():
    with capture_logs_with_contextvars() as cap_logs:
        logger = structlog.get_logger('testing')
        with ContextualizedLogging(context0='value0', context='value'):
            logger.info('context0 message')
            with ContextualizedLogging(context0='value0override'):
                logger.info('override context0 message')
            with ContextualizedLogging(context1='value1'):
                logger.info('context1 message', extra_value=1)
            logger.info('context0 message2')
    assert cap_logs == CONTEXT_RES
