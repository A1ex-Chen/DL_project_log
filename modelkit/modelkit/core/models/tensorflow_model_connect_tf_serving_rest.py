def connect_tf_serving_rest(model_name, host, port) ->None:
    try:
        for attempt in Retrying(**tf_serving_retry_policy('tf-serving-rest')):
            with attempt:
                response = requests.get(
                    f'http://{host}:{port}/v1/models/{model_name}')
                if response.status_code != 200:
                    raise TFServingError('Error connecting to TF serving')
    except requests.exceptions.ConnectionError:
        logger.error('Error connecting to TF serving')
        raise
