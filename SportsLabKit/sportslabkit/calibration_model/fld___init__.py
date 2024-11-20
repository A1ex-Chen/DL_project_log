def __init__(self, length_threshold=50, distance_threshold=50, canny_th1=50,
    canny_th2=150, canny_aperture_size=3, do_merge=True, dst_points=None):
    """Initialize the line-based calibrator with given parameters."""
    self.fld = cv2.ximgproc.createFastLineDetector(_length_threshold=self.
        length_threshold, _distance_threshold=self.distance_threshold,
        _canny_th1=self.canny_th1, _canny_th2=self.canny_th2,
        _canny_aperture_size=self.canny_aperture_size, _do_merge=self.do_merge)
    if dst_points is None:
        self.dst_points = np.array([[0, 0], [105, 0], [105, 68], [0, 68]],
            dtype=np.float32)
