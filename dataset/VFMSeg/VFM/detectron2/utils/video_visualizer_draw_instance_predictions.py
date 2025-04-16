def draw_instance_predictions(self, frame, predictions):
    """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
    frame_visualizer = Visualizer(frame, self.metadata)
    num_instances = len(predictions)
    if num_instances == 0:
        return frame_visualizer.output
    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has(
        'pred_boxes') else None
    scores = predictions.scores if predictions.has('scores') else None
    classes = predictions.pred_classes.numpy() if predictions.has(
        'pred_classes') else None
    keypoints = predictions.pred_keypoints if predictions.has('pred_keypoints'
        ) else None
    colors = predictions.COLOR if predictions.has('COLOR') else [None] * len(
        predictions)
    periods = predictions.ID_period if predictions.has('ID_period') else None
    period_threshold = self.metadata.get('period_threshold', 0)
    visibilities = [True] * len(predictions) if periods is None else [(x >
        period_threshold) for x in periods]
    if predictions.has('pred_masks'):
        masks = predictions.pred_masks
    else:
        masks = None
    if not predictions.has('COLOR'):
        if predictions.has('ID'):
            colors = self._assign_colors_by_id(predictions)
        else:
            detected = [_DetectedInstance(classes[i], boxes[i], mask_rle=
                None, color=colors[i], ttl=8) for i in range(num_instances)]
            colors = self._assign_colors(detected)
    labels = _create_text_labels(classes, scores, self.metadata.get(
        'thing_classes', None))
    if self._instance_mode == ColorMode.IMAGE_BW:
        frame_visualizer.output.reset_image(frame_visualizer.
            _create_grayscale_image((masks.any(dim=0) > 0).numpy() if masks
             is not None else None))
        alpha = 0.3
    else:
        alpha = 0.5
    labels = None if labels is None else [y[0] for y in filter(lambda x: x[
        1], zip(labels, visibilities))]
    assigned_colors = None if colors is None else [y[0] for y in filter(lambda
        x: x[1], zip(colors, visibilities))]
    frame_visualizer.overlay_instances(boxes=None if masks is not None else
        boxes[visibilities], masks=None if masks is None else masks[
        visibilities], labels=labels, keypoints=None if keypoints is None else
        keypoints[visibilities], assigned_colors=assigned_colors, alpha=alpha)
    return frame_visualizer.output
