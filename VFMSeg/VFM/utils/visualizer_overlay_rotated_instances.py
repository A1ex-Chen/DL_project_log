def overlay_rotated_instances(self, boxes=None, labels=None,
    assigned_colors=None):
    """
        Args:
            boxes (ndarray): an Nx5 numpy array of
                (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image.
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
    num_instances = len(boxes)
    if assigned_colors is None:
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range
            (num_instances)]
    if num_instances == 0:
        return self.output
    if boxes is not None:
        areas = boxes[:, 2] * boxes[:, 3]
    sorted_idxs = np.argsort(-areas).tolist()
    boxes = boxes[sorted_idxs]
    labels = [labels[k] for k in sorted_idxs] if labels is not None else None
    colors = [assigned_colors[idx] for idx in sorted_idxs]
    for i in range(num_instances):
        self.draw_rotated_box_with_label(boxes[i], edge_color=colors[i],
            label=labels[i] if labels is not None else None)
    return self.output
