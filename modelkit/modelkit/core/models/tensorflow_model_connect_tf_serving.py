def connect_tf_serving(model_name, host, port, mode):
    logger.info('Connecting to tensorflow serving', tf_serving_host=host,
        port=port, model_name=model_name, mode=mode)
    if mode == 'grpc':
        return connect_tf_serving_grpc(model_name, host, port)
    elif mode == 'rest':
        return connect_tf_serving_rest(model_name, host, port)
