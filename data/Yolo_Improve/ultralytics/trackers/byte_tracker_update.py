def update(self, results, img=None):
    """Updates object tracker with new detections and returns tracked object bounding boxes."""
    self.frame_id += 1
    activated_stracks = []
    refind_stracks = []
    lost_stracks = []
    removed_stracks = []
    scores = results.conf
    bboxes = results.xywhr if hasattr(results, 'xywhr') else results.xywh
    bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)],
        axis=-1)
    cls = results.cls
    remain_inds = scores >= self.args.track_high_thresh
    inds_low = scores > self.args.track_low_thresh
    inds_high = scores < self.args.track_high_thresh
    inds_second = inds_low & inds_high
    dets_second = bboxes[inds_second]
    dets = bboxes[remain_inds]
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]
    cls_keep = cls[remain_inds]
    cls_second = cls[inds_second]
    detections = self.init_track(dets, scores_keep, cls_keep, img)
    unconfirmed = []
    tracked_stracks = []
    for track in self.tracked_stracks:
        if not track.is_activated:
            unconfirmed.append(track)
        else:
            tracked_stracks.append(track)
    strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
    self.multi_predict(strack_pool)
    if hasattr(self, 'gmc') and img is not None:
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)
    dists = self.get_dists(strack_pool, detections)
    matches, u_track, u_detection = matching.linear_assignment(dists,
        thresh=self.args.match_thresh)
    for itracked, idet in matches:
        track = strack_pool[itracked]
        det = detections[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    detections_second = self.init_track(dets_second, scores_second,
        cls_second, img)
    r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].
        state == TrackState.Tracked]
    dists = matching.iou_distance(r_tracked_stracks, detections_second)
    matches, u_track, u_detection_second = matching.linear_assignment(dists,
        thresh=0.5)
    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections_second[idet]
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated_stracks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    for it in u_track:
        track = r_tracked_stracks[it]
        if track.state != TrackState.Lost:
            track.mark_lost()
            lost_stracks.append(track)
    detections = [detections[i] for i in u_detection]
    dists = self.get_dists(unconfirmed, detections)
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists,
        thresh=0.7)
    for itracked, idet in matches:
        unconfirmed[itracked].update(detections[idet], self.frame_id)
        activated_stracks.append(unconfirmed[itracked])
    for it in u_unconfirmed:
        track = unconfirmed[it]
        track.mark_removed()
        removed_stracks.append(track)
    for inew in u_detection:
        track = detections[inew]
        if track.score < self.args.new_track_thresh:
            continue
        track.activate(self.kalman_filter, self.frame_id)
        activated_stracks.append(track)
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:
            track.mark_removed()
            removed_stracks.append(track)
    self.tracked_stracks = [t for t in self.tracked_stracks if t.state ==
        TrackState.Tracked]
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks,
        activated_stracks)
    self.tracked_stracks = self.joint_stracks(self.tracked_stracks,
        refind_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.
        tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = self.sub_stracks(self.lost_stracks, self.
        removed_stracks)
    self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(
        self.tracked_stracks, self.lost_stracks)
    self.removed_stracks.extend(removed_stracks)
    if len(self.removed_stracks) > 1000:
        self.removed_stracks = self.removed_stracks[-999:]
    return np.asarray([x.result for x in self.tracked_stracks if x.
        is_activated], dtype=np.float32)
