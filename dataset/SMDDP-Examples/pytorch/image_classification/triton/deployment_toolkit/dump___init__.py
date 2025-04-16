def __init__(self, output_dir, compress=False):
    self._output_dir = Path(output_dir)
    self._items_cache: Dict[str, Dict[str, np.ndarray]] = {}
    self._items_counters: Dict[str, int] = {}
    self._flush_threshold_b = FLUSH_THRESHOLD_B
    self._compress = compress
