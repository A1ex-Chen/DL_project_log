def connect_tf_serving_grpc(model_name, host, port
    ) ->'prediction_service_pb2_grpc.PredictionServiceStub':
    try:
        for attempt in Retrying(**tf_serving_retry_policy('tf-serving-grpc')):
            with attempt:
                channel = grpc.insecure_channel(f'{host}:{port}', [(
                    'grpc.lb_policy_name', 'round_robin')])
                stub = prediction_service_pb2_grpc.PredictionServiceStub(
                    channel)
                r = GetModelMetadataRequest()
                r.model_spec.name = model_name
                r.metadata_field.append('signature_def')
                answ = stub.GetModelMetadata(r, 1)
                version = answ.model_spec.version.value
                if version != 1:
                    raise TFServingError(f'Bad model version: {version}!=1')
                return stub
    except grpc.RpcError:
        logger.error('Error connecting to TF serving')
        raise
