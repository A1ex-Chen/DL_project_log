def log_after_retry(retry_state):
    logger.info('Retrying', fun=retry_state.fn.__name__, attempt_number=
        retry_state.attempt_number, wait_time=retry_state.outcome_timestamp -
        retry_state.start_time)
