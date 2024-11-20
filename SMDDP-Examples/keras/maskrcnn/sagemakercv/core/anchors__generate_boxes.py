def _generate_boxes(self):
    """Generates multiscale anchor boxes."""
    boxes = _generate_anchor_boxes(self.image_size, self.anchor_scale, self
        .config)
    boxes = tf.convert_to_tensor(value=boxes, dtype=tf.float32)
    return boxes
