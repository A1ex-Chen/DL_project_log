def hit(self, detection: 'Detection', period: int=1):
    """Update tracked object with a new detection

        Parameters
        ----------
        detection : Detection
            the new detection matched to this tracked object
        period : int, optional
            frames corresponding to the period of time since last update.
        """
    self._conditionally_add_to_past_detections(detection)
    self.last_detection = detection
    self.hit_counter = min(self.hit_counter + 2 * period, self.hit_counter_max)
    if self.is_initializing and self.hit_counter > self.initialization_delay:
        self.is_initializing = False
        self._acquire_ids()
    if detection.scores is not None:
        assert len(detection.scores.shape) == 1
        points_over_threshold_mask = (detection.scores > self.
            detection_threshold)
        matched_sensors_mask = np.array([((m,) * self.dim_points) for m in
            points_over_threshold_mask]).flatten()
        H_pos = np.diag(matched_sensors_mask).astype(float)
        self.point_hit_counter[points_over_threshold_mask] += 2 * period
    else:
        points_over_threshold_mask = np.array([True] * self.num_points)
        H_pos = np.identity(self.num_points * self.dim_points)
        self.point_hit_counter += 2 * period
    self.point_hit_counter[self.point_hit_counter >= self.
        pointwise_hit_counter_max] = self.pointwise_hit_counter_max
    self.point_hit_counter[self.point_hit_counter < 0] = 0
    H_vel = np.zeros(H_pos.shape)
    H = np.hstack([H_pos, H_vel])
    self.filter.update(np.expand_dims(detection.absolute_points.flatten(), 
        0).T, None, H)
    detected_at_least_once_mask = np.array([((m,) * self.dim_points) for m in
        self.detected_at_least_once_points]).flatten()
    now_detected_mask = np.hstack((points_over_threshold_mask,) * self.
        dim_points).flatten()
    first_detection_mask = np.logical_and(now_detected_mask, np.logical_not
        (detected_at_least_once_mask))
    self.filter.x[:self.dim_z][first_detection_mask] = np.expand_dims(detection
        .absolute_points.flatten(), 0).T[first_detection_mask]
    self.filter.x[self.dim_z:][np.logical_not(detected_at_least_once_mask)] = 0
    self.detected_at_least_once_points = np.logical_or(self.
        detected_at_least_once_points, points_over_threshold_mask)
