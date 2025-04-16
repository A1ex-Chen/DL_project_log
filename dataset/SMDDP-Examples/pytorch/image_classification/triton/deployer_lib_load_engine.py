def load_engine(self, engine_filepath):
    """ loads a trt engine from engine_filepath, returns it """
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_filepath, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
