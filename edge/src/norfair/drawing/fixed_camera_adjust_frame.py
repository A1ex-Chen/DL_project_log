def adjust_frame(self, frame: np.ndarray, coord_transformation:
    TranslationTransformation) ->np.ndarray:
    """
        Render scaled up frame.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame.
        coord_transformation : TranslationTransformation
            The coordinate transformation as returned by the [`MotionEstimator`][norfair.camera_motion.MotionEstimator]

        Returns
        -------
        np.ndarray
            The new bigger frame with the original frame drawn on it.
        """
    if self._background is None:
        original_size = frame.shape[1], frame.shape[0]
        scaled_size = tuple((np.array(original_size) * np.array(self.scale)
            ).round().astype(int))
        self._background = np.zeros([scaled_size[1], scaled_size[0], frame.
            shape[-1]], frame.dtype)
    else:
        self._background = (self._background * self._attenuation_factor
            ).astype(frame.dtype)
    top_left = np.array(self._background.shape[:2]) // 2 - np.array(frame.
        shape[:2]) // 2
    top_left = coord_transformation.rel_to_abs(top_left[::-1]).round().astype(
        int)[::-1]
    background_y0, background_y1 = top_left[0], top_left[0] + frame.shape[0]
    background_x0, background_x1 = top_left[1], top_left[1] + frame.shape[1]
    background_size_y, background_size_x = self._background.shape[:2]
    frame_y0, frame_y1, frame_x0, frame_x1 = 0, frame.shape[0], 0, frame.shape[
        1]
    if (background_y0 < 0 or background_x0 < 0 or background_y1 >
        background_size_y or background_x1 > background_size_x):
        warn_once(
            'moving_camera_scale is not enough to cover the range of camera movement, frame will be cropped'
            )
        frame_y0 = max(-background_y0, 0)
        frame_x0 = max(-background_x0, 0)
        frame_y1 = max(min(background_size_y - background_y0, background_y1 -
            background_y0), 0)
        frame_x1 = max(min(background_size_x - background_x0, background_x1 -
            background_x0), 0)
        background_y0 = max(background_y0, 0)
        background_x0 = max(background_x0, 0)
        background_y1 = max(background_y1, 0)
        background_x1 = max(background_x1, 0)
    self._background[background_y0:background_y1, background_x0:
        background_x1, :] = frame[frame_y0:frame_y1, frame_x0:frame_x1, :]
    return self._background
