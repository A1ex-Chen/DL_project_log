def draw_and_connect_keypoints(self, keypoints):
    """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisImage): image object with visualizations.
        """
    visible = {}
    keypoint_names = self.metadata.get('keypoint_names')
    for idx, keypoint in enumerate(keypoints):
        x, y, prob = keypoint
        if prob > self.keypoint_threshold:
            self.draw_circle((x, y), color=_RED)
            if keypoint_names:
                keypoint_name = keypoint_names[idx]
                visible[keypoint_name] = x, y
    if self.metadata.get('keypoint_connection_rules'):
        for kp0, kp1, color in self.metadata.keypoint_connection_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                color = tuple(x / 255.0 for x in color)
                self.draw_line([x0, x1], [y0, y1], color=color)
    try:
        ls_x, ls_y = visible['left_shoulder']
        rs_x, rs_y = visible['right_shoulder']
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    except KeyError:
        pass
    else:
        nose_x, nose_y = visible.get('nose', (None, None))
        if nose_x is not None:
            self.draw_line([nose_x, mid_shoulder_x], [nose_y,
                mid_shoulder_y], color=_RED)
        try:
            lh_x, lh_y = visible['left_hip']
            rh_x, rh_y = visible['right_hip']
        except KeyError:
            pass
        else:
            mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
            self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y,
                mid_shoulder_y], color=_RED)
    return self.output
