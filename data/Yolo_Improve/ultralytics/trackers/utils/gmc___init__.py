def __init__(self, method: str='sparseOptFlow', downscale: int=2) ->None:
    """
        Initialize a video tracker with specified parameters.

        Args:
            method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
            downscale (int): Downscale factor for processing frames.
        """
    super().__init__()
    self.method = method
    self.downscale = max(1, downscale)
    if self.method == 'orb':
        self.detector = cv2.FastFeatureDetector_create(20)
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif self.method == 'sift':
        self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=
            0.02, edgeThreshold=20)
        self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold
            =0.02, edgeThreshold=20)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
    elif self.method == 'ecc':
        number_of_iterations = 5000
        termination_eps = 1e-06
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            number_of_iterations, termination_eps)
    elif self.method == 'sparseOptFlow':
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01,
            minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04)
    elif self.method in {'none', 'None', None}:
        self.method = None
    else:
        raise ValueError(f'Error: Unknown GMC method:{method}')
    self.prevFrame = None
    self.prevKeyPoints = None
    self.prevDescriptors = None
    self.initializedFirstFrame = False
