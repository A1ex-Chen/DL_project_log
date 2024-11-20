def calculate_speed(self, trk_id, track):
    """
        Calculates the speed of an object.

        Args:
            trk_id (int): Object track id.
            track (list): Tracking history for drawing tracks path.
        """
    if not self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]:
        return
    if self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1
        ] < self.reg_pts[1][1] + self.spdl_dist_thresh:
        direction = 'known'
    elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1
        ] < self.reg_pts[0][1] + self.spdl_dist_thresh:
        direction = 'known'
    else:
        direction = 'unknown'
    if self.trk_previous_times.get(trk_id
        ) != 0 and direction != 'unknown' and trk_id not in self.trk_idslist:
        self.trk_idslist.append(trk_id)
        time_difference = time() - self.trk_previous_times[trk_id]
        if time_difference > 0:
            dist_difference = np.abs(track[-1][1] - self.
                trk_previous_points[trk_id][1])
            speed = dist_difference / time_difference
            self.dist_data[trk_id] = speed
    self.trk_previous_times[trk_id] = time()
    self.trk_previous_points[trk_id] = track[-1]
