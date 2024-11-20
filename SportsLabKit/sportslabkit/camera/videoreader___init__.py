def __init__(self, filename: str, threaded=False, queue_size=10):
    """Open video in filename."""
    self._filename = filename
    self._vc = cv2.VideoCapture(str(self._filename))
    self.threaded = threaded
    self.stopped = False
    self.q: Queue = Queue(maxsize=queue_size)
    if threaded:
        t = threading.Thread(target=self.read_thread)
        t.daemon = True
        t.start()
