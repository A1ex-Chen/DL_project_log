def test_MotionVisualMatchingFunction_no_gates(self):
    matching_fn = MotionVisualMatchingFunction(motion_metric_gate=np.inf,
        visual_metric_gate=np.inf)
    matches = matching_fn(trackers=trackers, detections=dets)
    self.assertEqual(type(matches), np.ndarray)
    self.assertEqual(matches.shape, (2, 2))
    matches = matches.tolist()
    self.assertListEqual(matches[0], [0, 1])
    self.assertListEqual(matches[1], [1, 0])
