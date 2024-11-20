def update(self, frame: np.ndarray, mask: np.ndarray=None
    ) ->CoordinatesTransformation:
    """
        Estimate camera motion for each frame

        Parameters
        ----------
        frame : np.ndarray
            The frame.
        mask : np.ndarray, optional
            An optional mask to avoid areas of the frame when sampling the corner.
            Must be an array of shape `(frame.shape[0], frame.shape[1])`, dtype same as frame,
            and values in {0, 1}.

            In general, the estimation will work best when it samples many points from the background;
            with that intention, this parameters is usefull for masking out the detections/tracked objects,
            forcing the MotionEstimator ignore the moving objects.
            Can be used to mask static areas of the image, such as score overlays in sport transmisions or
            timestamps in security cameras.

        Returns
        -------
        CoordinatesTransformation
            The CoordinatesTransformation that can transform coordinates on this frame to absolute coordinates
            or vice versa.
        """
    self.gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if self.gray_prvs is None:
        self.gray_prvs = self.gray_next
        self.prev_mask = mask
    curr_pts, self.prev_pts = _get_sparse_flow(self.gray_next, self.
        gray_prvs, self.prev_pts, self.max_points, self.min_distance, self.
        block_size, self.prev_mask, quality_level=self.quality_level)
    if self.draw_flow:
        for curr, prev in zip(curr_pts, self.prev_pts):
            c = tuple(curr.astype(int).ravel())
            p = tuple(prev.astype(int).ravel())
            cv2.line(frame, c, p, self.flow_color, 2)
            cv2.circle(frame, c, 3, self.flow_color, -1)
    update_prvs, coord_transformations = self.transformations_getter(curr_pts,
        self.prev_pts)
    if update_prvs:
        self.gray_prvs = self.gray_next
        self.prev_pts = None
        self.prev_mask = mask
    return coord_transformations
