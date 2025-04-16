def __init__(self, paths: list, num_timed_runs=100, num_warmup_runs=10,
    min_time=60, imgsz=640, half=True, trt=True, device=None):
    """
        Initialize the ProfileModels class for profiling models.

        Args:
            paths (list): List of paths of the models to be profiled.
            num_timed_runs (int, optional): Number of timed runs for the profiling. Default is 100.
            num_warmup_runs (int, optional): Number of warmup runs before the actual profiling starts. Default is 10.
            min_time (float, optional): Minimum time in seconds for profiling a model. Default is 60.
            imgsz (int, optional): Size of the image used during profiling. Default is 640.
            half (bool, optional): Flag to indicate whether to use half-precision floating point for profiling.
            trt (bool, optional): Flag to indicate whether to profile using TensorRT. Default is True.
            device (torch.device, optional): Device used for profiling. If None, it is determined automatically.
        """
    self.paths = paths
    self.num_timed_runs = num_timed_runs
    self.num_warmup_runs = num_warmup_runs
    self.min_time = min_time
    self.imgsz = imgsz
    self.half = half
    self.trt = trt
    self.device = device or torch.device(0 if torch.cuda.is_available() else
        'cpu')
