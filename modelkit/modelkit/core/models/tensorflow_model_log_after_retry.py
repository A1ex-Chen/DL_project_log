def log_after_retry(name):
    return lambda retry_state: logger.info('Retrying TF serving connection',
        name=name, attempt_number=retry_state.attempt_number, wait_time=
        retry_state.outcome_timestamp - retry_state.start_time)
