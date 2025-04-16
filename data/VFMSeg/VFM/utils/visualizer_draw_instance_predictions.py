def draw_instance_predictions(self, predictions):
    """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
    boxes = predictions.pred_boxes if predictions.has('pred_boxes') else None
    scores = predictions.scores if predictions.has('scores') else None
    classes = predictions.pred_classes.tolist() if predictions.has(
        'pred_classes') else None
    labels = _create_text_labels(classes, scores, self.metadata.get(
        'thing_classes', None))
    keypoints = predictions.pred_keypoints if predictions.has('pred_keypoints'
        ) else None
    keep = (scores > 0.8).cpu()
    boxes = boxes[keep]
    scores = scores[keep]
    classes = np.array(classes)
    classes = classes[np.array(keep)]
    labels = np.array(labels)
    labels = labels[np.array(keep)]
    if predictions.has('pred_masks'):
        masks = np.asarray(predictions.pred_masks)
        masks = masks[np.array(keep)]
        masks = [GenericMask(x, self.output.height, self.output.width) for
            x in masks]
    else:
        masks = None
    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
        'thing_colors'):
        colors = [self._jitter([(x / 255) for x in self.metadata.
            thing_colors[c]]) for c in classes]
        alpha = 0.4
    else:
        colors = None
        alpha = 0.4
    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(self._create_grayscale_image((predictions.
            pred_masks.any(dim=0) > 0).numpy() if predictions.has(
            'pred_masks') else None))
        alpha = 0.3
    self.overlay_instances(masks=masks, boxes=boxes, labels=labels,
        keypoints=keypoints, assigned_colors=colors, alpha=alpha)
    return self.output
