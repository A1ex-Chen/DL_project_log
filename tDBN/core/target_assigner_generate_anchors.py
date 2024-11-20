def generate_anchors(self, feature_map_size):
    anchors_list = []
    matched_thresholds = [a.match_threshold for a in self._anchor_generators]
    unmatched_thresholds = [a.unmatch_threshold for a in self.
        _anchor_generators]
    match_list, unmatch_list = [], []
    for anchor_generator, match_thresh, unmatch_thresh in zip(self.
        _anchor_generators, matched_thresholds, unmatched_thresholds):
        anchors = anchor_generator.generate(feature_map_size)
        anchors = anchors.reshape([*anchors.shape[:3], -1, 7])
        anchors_list.append(anchors)
        num_anchors = np.prod(anchors.shape[:-1])
        match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
        unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.
            dtype))
    anchors = np.concatenate(anchors_list, axis=-2)
    matched_thresholds = np.concatenate(match_list, axis=0)
    unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
    return {'anchors': anchors, 'matched_thresholds': matched_thresholds,
        'unmatched_thresholds': unmatched_thresholds}
