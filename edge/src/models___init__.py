def __init__(self, weights='reid_fp32.trt'):
    LOGGER.info('Loading extracting model...')
    self.img_size = 192, 64
    f = open(weights, 'rb')
    self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    self.engine = self.runtime.deserialize_cuda_engine(f.read())
    self.context = self.engine.create_execution_context()
    self.output = np.empty([1, 224], dtype=np.float32)
    temp_batch = self.preprocess(np.random.randint(0, 255, (np.random.
        randint(20, 400), np.random.randint(20, 400), 3), 'uint8'))
    self.d_input = cuda.mem_alloc(1 * temp_batch.nbytes)
    self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
    self.bindings = [int(self.d_input), int(self.d_output)]
    self.stream = cuda.Stream()
