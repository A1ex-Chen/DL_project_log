def parse_record(self, record: tf.Tensor) ->Tuple[tf.Tensor, tf.Tensor]:
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {'image/encoded': tf.io.FixedLenFeature((), tf.
        string, ''), 'image/format': tf.io.FixedLenFeature((), tf.string,
        'jpeg'), 'image/class/label': tf.io.FixedLenFeature([], tf.int64, -
        1), 'image/class/text': tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(dtype=tf.int64)}
    parsed = tf.io.parse_single_example(record, keys_to_features)
    label = tf.reshape(parsed['image/class/label'], shape=[1])
    label = tf.cast(label, dtype=tf.int32)
    label -= 1
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image, label = self.preprocess(image_bytes, label)
    return image, label
