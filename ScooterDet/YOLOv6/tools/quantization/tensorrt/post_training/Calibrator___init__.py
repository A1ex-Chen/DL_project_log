def __init__(self, calibration_files=[], batch_size=32, input_shape=(3, 224,
    224), cache_file='calibration.cache', use_cv2=False):
    super().__init__()
    self.input_shape = input_shape
    self.cache_file = cache_file
    self.batch_size = batch_size
    self.batch = np.zeros((self.batch_size, *self.input_shape), dtype=np.
        float32)
    self.device_input = cuda.mem_alloc(self.batch.nbytes)
    self.files = calibration_files
    self.use_cv2 = use_cv2
    if len(self.files) % self.batch_size != 0:
        logger.info(
            'Padding # calibration files to be a multiple of batch_size {:}'
            .format(self.batch_size))
        self.files += calibration_files[len(calibration_files) % self.
            batch_size:self.batch_size]
    self.batches = self.load_batches()
    self.preprocess_func = preprocess_yolov6
