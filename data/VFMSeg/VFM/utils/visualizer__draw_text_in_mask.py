def _draw_text_in_mask(self, binary_mask, text, color):
    """
        Find proper places to draw text given a binary mask.
        """
    _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, 8)
    if stats[1:, -1].size == 0:
        return
    largest_component_id = np.argmax(stats[1:, -1]) + 1
    for cid in range(1, _num_cc):
        if cid == largest_component_id or stats[cid, -1
            ] > _LARGE_MASK_AREA_THRESH:
            center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
            self.draw_text(text, center, color=color)
