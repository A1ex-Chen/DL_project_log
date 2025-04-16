def __init__(self, dataset, batch: int, cache: str='') ->None:
    trt.IInt8Calibrator.__init__(self)
    self.dataset = dataset
    self.data_iter = iter(dataset)
    self.algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    self.batch = batch
    self.cache = Path(cache)
