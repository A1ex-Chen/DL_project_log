def overlay_instances(self, *, boxes=None, labels=None, masks=None,
    keypoints=None, assigned_colors=None, alpha=0.5):
    """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.
        Returns:
            output (VisImage): image object with visualizations.
        """
    num_instances = 0
    if boxes is not None:
        boxes = self._convert_boxes(boxes)
        num_instances = len(boxes)
    if masks is not None:
        masks = self._convert_masks(masks)
        if num_instances:
            assert len(masks) == num_instances
        else:
            num_instances = len(masks)
    if keypoints is not None:
        if num_instances:
            assert len(keypoints) == num_instances
        else:
            num_instances = len(keypoints)
        keypoints = self._convert_keypoints(keypoints)
    if labels is not None:
        assert len(labels) == num_instances
    if assigned_colors is None:
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range
            (num_instances)]
    if num_instances == 0:
        return self.output
    if boxes is not None and boxes.shape[1] == 5:
        return self.overlay_rotated_instances(boxes=boxes, labels=labels,
            assigned_colors=assigned_colors)
    areas = None
    if boxes is not None:
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    elif masks is not None:
        areas = np.asarray([x.area() for x in masks])
    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        boxes = boxes[sorted_idxs] if boxes is not None else None
        labels = [labels[k] for k in sorted_idxs
            ] if labels is not None else None
        masks = [masks[idx] for idx in sorted_idxs
            ] if masks is not None else None
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
        keypoints = keypoints[sorted_idxs] if keypoints is not None else None
    for i in range(num_instances):
        color = assigned_colors[i]
        if boxes is not None:
            self.draw_box(boxes[i], edge_color=color)
        if masks is not None:
            for segment in masks[i].polygons:
                self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)
        if labels is not None:
            if boxes is not None:
                x0, y0, x1, y1 = boxes[i]
                text_pos = x0, y0
                horiz_align = 'left'
            elif masks is not None:
                if len(masks[i].polygons) == 0:
                    continue
                x0, y0, x1, y1 = masks[i].bbox()
                text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                horiz_align = 'center'
            else:
                continue
            instance_area = (y1 - y0) * (x1 - x0)
            if (instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.
                scale or y1 - y0 < 40 * self.output.scale):
                if y1 >= self.output.height - 5:
                    text_pos = x1, y0
                else:
                    text_pos = x0, y1
            height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.
                output.width)
            lighter_color = self._change_color_brightness(color,
                brightness_factor=0.7)
            font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2
                ) * 0.5 * self._default_font_size
            self.draw_text(labels[i], text_pos, color=lighter_color,
                horizontal_alignment=horiz_align, font_size=font_size)
    if keypoints is not None:
        for keypoints_per_instance in keypoints:
            self.draw_and_connect_keypoints(keypoints_per_instance)
    return self.output
