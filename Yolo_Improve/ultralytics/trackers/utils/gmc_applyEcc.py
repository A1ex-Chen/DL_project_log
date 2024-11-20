def applyEcc(self, raw_frame: np.array) ->np.array:
    """
        Apply ECC algorithm to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed.

        Returns:
            (np.ndarray): Processed frame.

        Examples:
            >>> gmc = GMC()
            >>> gmc.applyEcc(np.array([[1, 2, 3], [4, 5, 6]]))
            array([[1, 2, 3],
                   [4, 5, 6]])
        """
    height, width, _ = raw_frame.shape
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    H = np.eye(2, 3, dtype=np.float32)
    if self.downscale > 1.0:
        frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
        frame = cv2.resize(frame, (width // self.downscale, height // self.
            downscale))
        width = width // self.downscale
        height = height // self.downscale
    if not self.initializedFirstFrame:
        self.prevFrame = frame.copy()
        self.initializedFirstFrame = True
        return H
    try:
        _, H = cv2.findTransformECC(self.prevFrame, frame, H, self.
            warp_mode, self.criteria, None, 1)
    except Exception as e:
        LOGGER.warning(
            f'WARNING: find transform failed. Set warp as identity {e}')
    return H
