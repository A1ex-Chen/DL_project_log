def tf_serving_retry_policy(name):
    return {'wait': wait_random_exponential(multiplier=1, min=4, max=20),
        'stop': stop_after_attempt(int(os.environ.get(
        'MODELKIT_TF_SERVING_ATTEMPTS', 10))), 'retry': retry_if_exception(
        retriable_error), 'after': log_after_retry(name), 'reraise': True}
