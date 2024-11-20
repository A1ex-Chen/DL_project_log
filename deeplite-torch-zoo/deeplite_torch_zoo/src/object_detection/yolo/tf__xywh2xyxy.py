@staticmethod
def _xywh2xyxy(xywh):
    x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
    return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
