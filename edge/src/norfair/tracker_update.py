def update(self, detections: Optional[List['Detection']]=None, period: int=
    1, coord_transformations: Optional[CoordinatesTransformation]=None) ->List[
    'TrackedObject']:
    """
        Process detections found in each frame.

        The detections can be matched to previous tracked objects or new ones will be created
        according to the configuration of the Tracker.
        The currently alive and initialized tracked objects are returned

        Parameters
        ----------
        detections : Optional[List[Detection]], optional
            A list of [`Detection`][norfair.tracker.Detection] which represent the detections found in the current frame being processed.

            If no detections have been found in the current frame, or the user is purposely skipping frames to improve video processing time,
            this argument should be set to None or ignored, as the update function is needed to advance the state of the Kalman Filters inside the tracker.
        period : int, optional
            The user can chose not to run their detector on all frames, so as to process video faster.
            This parameter sets every how many frames the detector is getting ran,
            so that the tracker is aware of this situation and can handle it properly.

            This argument can be reset on each frame processed,
            which is useful if the user is dynamically changing how many frames the detector is skipping on a video when working in real-time.
        coord_transformations: Optional[CoordinatesTransformation]
            The coordinate transformation calculated by the [MotionEstimator][norfair.camera_motion.MotionEstimator].

        Returns
        -------
        List[TrackedObject]
            The list of active tracked objects.
        """
    if coord_transformations is not None:
        for det in detections:
            det.update_coordinate_transformation(coord_transformations)
    alive_objects = []
    dead_objects = []
    if self.reid_hit_counter_max is None:
        self.tracked_objects = [o for o in self.tracked_objects if o.
            hit_counter_is_positive]
        alive_objects = self.tracked_objects
    else:
        tracked_objects = []
        for o in self.tracked_objects:
            if o.reid_hit_counter_is_positive:
                tracked_objects.append(o)
                if o.hit_counter_is_positive:
                    alive_objects.append(o)
                else:
                    dead_objects.append(o)
        self.tracked_objects = tracked_objects
    for obj in self.tracked_objects:
        obj.tracker_step()
        obj.update_coordinate_transformation(coord_transformations)
    unmatched_detections, _, unmatched_init_trackers = (self.
        _update_objects_in_place(self.distance_function, self.
        distance_threshold, [o for o in alive_objects if not o.
        is_initializing], detections, period))
    unmatched_detections, _, unmatched_init_trackers = (self.
        _update_objects_in_place(self.distance_function2, 0.7,
        unmatched_init_trackers, unmatched_detections, period))
    unmatched_detections, matched_not_init_trackers, _ = (self.
        _update_objects_in_place(self.distance_function2, 0.7, [o for o in
        alive_objects if o.is_initializing], unmatched_detections, period))
    if self.reid_distance_function is not None:
        _, _, _ = self._update_objects_in_place(self.reid_distance_function,
            self.reid_distance_threshold, unmatched_init_trackers +
            dead_objects, matched_not_init_trackers, period)
    for detection in unmatched_detections:
        self.tracked_objects.append(self._obj_factory.create(
            initial_detection=detection, hit_counter_max=self.
            hit_counter_max, initialization_delay=self.initialization_delay,
            pointwise_hit_counter_max=self.pointwise_hit_counter_max,
            detection_threshold=self.detection_threshold, period=period,
            filter_factory=self.filter_factory, past_detections_length=self
            .past_detections_length, reid_hit_counter_max=self.
            reid_hit_counter_max, coord_transformations=coord_transformations))
    return self.get_active_objects()
