def get_config(self, path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(path, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    return config
