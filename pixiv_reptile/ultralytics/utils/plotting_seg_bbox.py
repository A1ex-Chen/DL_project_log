def seg_bbox(self, mask, mask_color=(255, 0, 255), det_label=None,
    track_label=None):
    """
        Function for drawing segmented object in bounding box shape.

        Args:
            mask (list): masks data list for instance segmentation area plotting
            mask_color (tuple): mask foreground color
            det_label (str): Detection label text
            track_label (str): Tracking label text
        """
    cv2.polylines(self.im, [np.int32([mask])], isClosed=True, color=
        mask_color, thickness=2)
    label = f'Track ID: {track_label}' if track_label else det_label
    text_size, _ = cv2.getTextSize(label, 0, self.sf, self.tf)
    cv2.rectangle(self.im, (int(mask[0][0]) - text_size[0] // 2 - 10, int(
        mask[0][1]) - text_size[1] - 10), (int(mask[0][0]) + text_size[0] //
        2 + 10, int(mask[0][1] + 10)), mask_color, -1)
    cv2.putText(self.im, label, (int(mask[0][0]) - text_size[0] // 2, int(
        mask[0][1])), 0, self.sf, (255, 255, 255), self.tf)
